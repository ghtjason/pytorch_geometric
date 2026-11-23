from typing import Optional

import torch
from torch import Tensor
from torch.nn import (
    GELU,
    BatchNorm1d,
    Dropout,
    LayerNorm,
    MultiheadAttention,
    Sequential,
)

from torch_geometric.nn.dense.linear import Linear


class RelGTLayer(torch.nn.Module):
    r"""The local transformer layer from the `"Relational Graph Transformer"
    <https://arxiv.org/abs/2505.10960>`_ paper.

    Lets the :math:`K` tokens of each seed node attend to each other via
    all-pair multi-head self-attention, followed by a feed-forward network,
    both wrapped in pre-normalized residual connections:

    .. math::
        \mathbf{X} &\leftarrow \mathbf{X} + \mathrm{Dropout} \left(
        \mathrm{Attention} \left( \mathrm{LayerNorm}(\mathbf{X}) \right)
        \right)

        \mathbf{X} &\leftarrow \mathbf{X} + \mathrm{FFN} \left(
        \mathrm{LayerNorm}(\mathbf{X}) \right)

    where :math:`\mathrm{FFN}` applies a two-layer MLP with GELU activation
    and dropout, wrapped in :class:`~torch.nn.BatchNorm1d` layers over all
    tokens of the mini-batch, as in the reference implementation.
    The global attention to centroids of the paper is implemented separately
    in :class:`~torch_geometric.contrib.nn.attention.GlobalCentroidAttention`.

    .. note::

        The reference implementation uses a dropout probability of
        :obj:`0.3`.

    Args:
        channels (int): Size :math:`d` of each token.
        heads (int, optional): The number of attention heads.
            Needs to divide :obj:`channels`. (default: :obj:`1`)
        dropout (float, optional): Dropout probability of the attention and
            feed-forward outputs. (default: :obj:`0.0`)
        attn_dropout (float, optional): Dropout probability of the attention
            weights. (default: :obj:`0.0`)
        ffn_channels (int, optional): The hidden size of the feed-forward
            network. If set to :obj:`None`, will use :obj:`2 * channels`.
            (default: :obj:`None`)

    Shapes:
        - **input:** tokens :math:`(B, K, d)`
        - **output:** tokens :math:`(B, K, d)`
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_channels: Optional[int] = None,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"'channels' (got {channels}) must be divisible "
                             f"by 'heads' (got {heads})")

        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_channels = ffn_channels or 2 * channels

        self.attn_norm = LayerNorm(channels)
        self.attn = MultiheadAttention(channels, heads, dropout=attn_dropout,
                                       batch_first=True)
        self.attn_out_dropout = Dropout(dropout)

        self.ffn_norm = LayerNorm(channels)
        self.ffn_in_norm = BatchNorm1d(channels)
        self.ffn = Sequential(
            Linear(channels, self.ffn_channels),
            GELU(),
            Dropout(dropout),
            Linear(self.ffn_channels, channels),
            Dropout(dropout),
        )
        self.ffn_out_norm = BatchNorm1d(channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.attn_norm.reset_parameters()
        self.attn._reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn_in_norm.reset_parameters()
        self.ffn[0].reset_parameters()
        self.ffn[3].reset_parameters()
        self.ffn_out_norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The tokens of each seed node, of shape
                :obj:`[B, K, channels]`.
        """
        if x.dim() != 3 or x.size(-1) != self.channels:
            raise ValueError(f"Expected 'x' to have shape "
                             f"[B, K, {self.channels}] "
                             f"(got {tuple(x.size())})")

        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_out_dropout(h)

        # Batch normalization is applied over all `B * K` tokens:
        h = self.ffn_norm(x).view(-1, self.channels)
        h = self.ffn_out_norm(self.ffn(self.ffn_in_norm(h)))
        return x + h.view(x.size())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'heads={self.heads})')
