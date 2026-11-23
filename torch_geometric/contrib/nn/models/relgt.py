from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    GELU,
    BatchNorm1d,
    Dropout,
    LayerNorm,
    ModuleList,
    Sequential,
)

from torch_geometric.contrib.nn.attention import (
    GlobalCentroidAttention,
    RelGTLayer,
)
from torch_geometric.contrib.nn.encoding import RelGTTokenizer
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models import MLP


class RelGT(torch.nn.Module):
    r"""The Relational Graph Transformer (RelGT) from the `"Relational Graph
    Transformer" <https://arxiv.org/abs/2505.10960>`_ paper, a graph
    transformer for relational entity graphs.

    Each seed node :math:`v_i` is represented by :math:`K` tokens sampled
    from its (temporal) neighborhood, *e.g.*, via
    :class:`~torch_geometric.contrib.transforms.RelGTTokenSampler`.
    The model combines

    1. a :class:`~torch_geometric.contrib.nn.encoding.RelGTTokenizer`, which
       embeds the features, node type, hop distance, relative time and
       subgraph positional encoding of every token,
    2. a local module of :obj:`num_layers`
       :class:`~torch_geometric.contrib.nn.attention.RelGTLayer` layers
       followed by an attention readout of the :math:`K` output tokens
       :math:`\mathbf{z}_0, \ldots, \mathbf{z}_{K-1}`, where
       :math:`\mathbf{z}_0` belongs to the seed node:

       .. math::
           \mathbf{h}_{\textrm{local}}(v_i) = \mathbf{z}_0 + \sum_{j=1}^{K-1}
           \mathrm{softmax}_j \left( \mathbf{w}^{\top}
           [\mathbf{z}_0 \, \Vert \, \mathbf{z}_j] \right) \mathbf{z}_j,

    3. a global module, in which the input token of the seed node attends to
       centroids of all seed nodes via
       :class:`~torch_geometric.contrib.nn.attention.GlobalCentroidAttention`,
       and
    4. a feed-forward network and prediction head applied to
       :math:`[\mathbf{h}_{\textrm{local}}(v_i) \, \Vert \,
       \mathbf{h}_{\textrm{global}}(v_i)]`.

    Following the reference implementation, the outputs of the local and
    global module are normalized via :class:`~torch.nn.LayerNorm`, and the
    feed-forward network is wrapped in :class:`~torch.nn.BatchNorm1d` layers.
    In contrast to the reference implementation, no additional linear
    projection is applied to the tokens before the local module, since the
    tokenizer already ends in a linear layer.

    .. note::

        The default subgraph positional encoding draws new random node
        features in every forward pass, including in evaluation mode, such
        that repeated predictions differ slightly unless a random seed is
        fixed.

    Args:
        channels (int): Size :math:`d` of each token and hidden
            representation. Needs to be at least 4 if :obj:`pe_encoder` is
            :obj:`None`.
        out_channels (int): Size of each output.
        node_types (List[str]): The node types (*i.e.*, table names), in the
            order used by the node type IDs of the tokens.
        feature_encoder (torch.nn.Module): A module mapping a dictionary of
            per-node-type inputs to a dictionary of features of shape
            :obj:`[num_nodes_of_type, channels]`, such as the
            :class:`HeteroEncoder` of the RelBench pipeline.
        num_layers (int, optional): The number of local transformer layers.
            (default: :obj:`1`)
        heads (int, optional): The number of attention heads.
            (default: :obj:`1`)
        num_centroids (int, optional): The number of centroids of the global
            module. (default: :obj:`4096`)
        max_hops (int, optional): The maximum hop distance of sampled
            neighbors. (default: :obj:`2`)
        dropout (float, optional): Dropout probability of the feed-forward
            networks and attention outputs. (default: :obj:`0.0`)
        attn_dropout (float, optional): Dropout probability of the attention
            weights. (default: :obj:`0.0`)
        pe_encoder (torch.nn.Module, optional): The positional encoder of the
            tokenizer. If set to :obj:`None`, will use
            :class:`~torch_geometric.contrib.nn.encoding.SubgraphGNNPE`.
            (default: :obj:`None`)
        global_attn (bool, optional): If set to :obj:`False`, will not use the
            global module. (default: :obj:`True`)
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        node_types: List[str],
        feature_encoder: torch.nn.Module,
        num_layers: int = 1,
        heads: int = 1,
        num_centroids: int = 4096,
        max_hops: int = 2,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        pe_encoder: Optional[torch.nn.Module] = None,
        global_attn: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"'num_layers' must be positive "
                             f"(got {num_layers})")

        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads

        self.tokenizer = RelGTTokenizer(channels, node_types, feature_encoder,
                                        max_hops=max_hops,
                                        pe_encoder=pe_encoder)

        self.layers = ModuleList([
            RelGTLayer(channels, heads=heads, dropout=dropout,
                       attn_dropout=attn_dropout) for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(channels)
        self.readout = Linear(2 * channels, 1)
        self.local_norm = LayerNorm(channels)

        self.global_attn: Optional[GlobalCentroidAttention] = None
        self.global_norm: Optional[LayerNorm] = None
        if global_attn:
            self.global_attn = GlobalCentroidAttention(
                channels, num_centroids=num_centroids, heads=heads,
                dropout=attn_dropout)
            self.global_norm = LayerNorm(channels)

        in_channels = 2 * channels if global_attn else channels
        self.ffn = Sequential(
            BatchNorm1d(in_channels),
            Linear(in_channels, 2 * channels),
            GELU(),
            Dropout(dropout),
            Linear(2 * channels, channels),
            Dropout(dropout),
            BatchNorm1d(channels),
        )
        self.head = MLP([channels, channels, out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.tokenizer.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.final_norm.reset_parameters()
        self.readout.reset_parameters()
        self.local_norm.reset_parameters()
        if self.global_attn is not None:
            self.global_attn.reset_parameters()
        if self.global_norm is not None:
            self.global_norm.reset_parameters()
        for module in self.ffn:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.head.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[str, Any],
        node_idx: Tensor,
        node_type: Tensor,
        hop: Tensor,
        rel_time: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            feat_dict (Dict[str, Any]): The per-node-type inputs to
                :obj:`feature_encoder`, *e.g.*, a dictionary holding a
                :class:`torch_frame.TensorFrame` for each node type.
            node_idx (torch.Tensor): The index of each token within its node
                type, of shape :obj:`[B, K]`, where the first token of every
                seed node is the seed node itself.
            node_type (torch.Tensor): The node type ID of each token, of shape
                :obj:`[B, K]`.
            hop (torch.Tensor): The hop distance of each token from its seed
                node, of shape :obj:`[B, K]`.
            rel_time (torch.Tensor): The time of each seed node minus the time
                of each of its tokens in seconds, of shape :obj:`[B, K]`.
            edge_index (torch.Tensor): The edge indices of the sampled
                subgraphs, where token :obj:`k` of seed node :obj:`b` is
                denoted by node index :obj:`b * K + k`.
        """
        x = self.tokenizer(feat_dict, node_idx, node_type, hop, rel_time,
                           edge_index)  # [B, K, channels]

        z = x
        for layer in self.layers:
            z = layer(z)
        z = self.final_norm(z)
        h = self.local_norm(self._readout(z))

        if self.global_attn is not None and self.global_norm is not None:
            h_global = self.global_norm(self.global_attn(x[:, 0]))
            h = torch.cat([h, h_global], dim=-1)

        return self.head(self.ffn(h))

    def _readout(self, z: Tensor) -> Tensor:
        r"""Aggregates the output tokens of each seed node, weighting the
        neighbor tokens via attention conditioned on the seed node token.
        """
        seed, nbr = z[:, 0], z[:, 1:]
        score = self.readout(
            torch.cat([seed.unsqueeze(1).expand_as(nbr), nbr], dim=-1))
        alpha = F.softmax(score, dim=1)  # [B, K - 1, 1]
        return seed + (alpha * nbr).sum(dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'{self.out_channels}, num_layers={self.num_layers}, '
                f'heads={self.heads}, '
                f'global_attn={self.global_attn is not None})')
