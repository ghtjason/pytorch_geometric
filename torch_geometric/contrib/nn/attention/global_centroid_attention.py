import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d

from torch_geometric.nn.dense.linear import Linear


class GlobalCentroidAttention(torch.nn.Module):
    r"""The global centroid attention from the `"Relational Graph
    Transformer" <https://arxiv.org/abs/2505.10960>`_ paper.

    Every query :math:`\mathbf{x}_i` attends to :math:`B` centroids
    :math:`\mathbf{c}_1, \ldots, \mathbf{c}_B` that summarize all queries seen
    during training, at a cost that does not depend on the size of the graph:

    .. math::
        \mathbf{h}_i = \sum_{b=1}^B \alpha_{i,b} \mathbf{W}_V \mathbf{c}_b,
        \qquad
        \alpha_{i,b} = \mathrm{softmax}_b \left(
        \frac{(\mathbf{W}_Q \mathbf{x}_i)^{\top} \mathbf{W}_K \mathbf{c}_b}
        {\sqrt{d / H}} + \log n_b \right),

    computed separately for each of the :math:`H` heads.
    Here, :math:`n_b` denotes the (exponential moving average of the) number
    of queries assigned to centroid :math:`\mathbf{c}_b`, such that attending
    to a centroid approximates attending to all queries it represents.

    The centroids are not learned via gradient descent.
    Instead, every forward pass in training mode performs one exponential
    moving average (EMA) :math:`k`-means step on the given queries, as in
    `"Neural Discrete Representation Learning"
    <https://arxiv.org/abs/1711.00937>`_:
    Queries are normalized via a :class:`~torch.nn.BatchNorm1d` layer (without
    affine parameters) and assigned to their closest centroid, before the
    cluster sizes :math:`n_b` and cluster sums :math:`\mathbf{s}_b` are updated
    via

    .. math::
        n_b \leftarrow \gamma n_b + (1 - \gamma) |\mathcal{A}_b|, \qquad
        \mathbf{s}_b \leftarrow \gamma \mathbf{s}_b + (1 - \gamma)
        \sum_{i \in \mathcal{A}_b} \mathbf{x}_i, \qquad
        \mathbf{c}_b \leftarrow \mathbf{s}_b / n_b,

    where :math:`\mathcal{A}_b` holds the queries assigned to centroid
    :math:`b` and :math:`\gamma` denotes the decay.
    Cluster sizes are Laplace-smoothed to stay positive.
    Keys and values are computed from the centroids mapped back to the
    original scale of the queries via the running statistics of the
    normalization layer.

    .. note::

        Mini-batches with a single query do not update the centroids, since
        their batch statistics are undefined.

    Args:
        channels (int): Size :math:`d` of each query.
        num_centroids (int, optional): The number of centroids :math:`B`.
            (default: :obj:`4096`)
        heads (int, optional): The number of attention heads :math:`H`.
            Needs to divide :obj:`channels`. (default: :obj:`1`)
        decay (float, optional): The decay :math:`\gamma` of the EMA
            :math:`k`-means update. (default: :obj:`0.99`)
        dropout (float, optional): Dropout probability of the attention
            weights. (default: :obj:`0.0`)

    Shapes:
        - **input:** queries :math:`(N, d)`
        - **output:** outputs :math:`(N, d)`
    """
    def __init__(
        self,
        channels: int,
        num_centroids: int = 4096,
        heads: int = 1,
        decay: float = 0.99,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"'channels' (got {channels}) must be divisible "
                             f"by 'heads' (got {heads})")
        if num_centroids < 1:
            raise ValueError(f"'num_centroids' must be positive "
                             f"(got {num_centroids})")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"'decay' must be in [0, 1) (got {decay})")

        self.channels = channels
        self.num_centroids = num_centroids
        self.heads = heads
        self.decay = decay
        self.dropout = dropout
        self.eps = 1e-5

        self.lin_query = Linear(channels, channels)
        self.lin_key = Linear(channels, channels)
        self.lin_value = Linear(channels, channels)
        self.norm = BatchNorm1d(channels, affine=False, eps=self.eps)

        self.register_buffer('centroids', torch.empty(num_centroids, channels))
        self.register_buffer('cluster_size', torch.empty(num_centroids))
        self.register_buffer('cluster_sum',
                             torch.empty(num_centroids, channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters and centroids of the module."""
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.norm.reset_parameters()
        torch.nn.init.normal_(self.centroids)
        # Start with every centroid representing a single query, such that
        # `cluster_sum / cluster_size` matches the initial centroids:
        self.cluster_size.fill_(1.0)
        self.cluster_sum.copy_(self.centroids)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The queries, of shape :obj:`[N, channels]`.
        """
        if x.dim() != 2 or x.size(-1) != self.channels:
            raise ValueError(f"Expected 'x' to have shape "
                             f"[*, {self.channels}] (got {tuple(x.size())})")

        H, C = self.heads, self.channels // self.heads

        # Map the centroids back to the original scale of the queries:
        std = (self.norm.running_var + self.eps).sqrt()
        centroids = self.centroids * std + self.norm.running_mean

        q = self.lin_query(x).view(-1, H, C)
        k = self.lin_key(centroids).view(-1, H, C)
        v = self.lin_value(centroids).view(-1, H, C)

        logits = torch.einsum('nhc,bhc->nhb', q, k) / math.sqrt(C)
        logits = logits + self.cluster_size.log().view(1, 1, -1)
        alpha = F.softmax(logits, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.einsum('nhb,bhc->nhc', alpha, v).reshape(-1, self.channels)

        if self.training:
            self._update(x.detach())

        return out

    @torch.no_grad()
    def _update(self, x: Tensor):
        r"""Performs one EMA :math:`k`-means step on the queries :obj:`x`."""
        if x.size(0) < 2:
            return

        x = self.norm(x)
        dist = (x.pow(2).sum(dim=-1, keepdim=True) -
                2 * x @ self.centroids.t() +
                self.centroids.pow(2).sum(dim=-1).view(1, -1))
        assignment = F.one_hot(dist.argmin(dim=-1), self.num_centroids)
        assignment = assignment.to(x.dtype)

        size = self.cluster_size * self.decay
        size += (1 - self.decay) * assignment.sum(dim=0)
        # Laplace smoothing keeps cluster sizes positive (and their sum):
        total = size.sum()
        size = (size + self.eps) / (total + self.num_centroids * self.eps)
        self.cluster_size.copy_(size * total)

        self.cluster_sum.mul_(self.decay)
        self.cluster_sum.add_((1 - self.decay) * assignment.t() @ x)
        self.centroids.copy_(self.cluster_sum / self.cluster_size.view(-1, 1))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'num_centroids={self.num_centroids}, heads={self.heads})')
