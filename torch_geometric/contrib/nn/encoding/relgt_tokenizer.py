from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.contrib.nn.encoding.subgraph_gnn_pe import SubgraphGNNPE
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.encoding import PositionalEncoding

SECONDS_PER_DAY = 60 * 60 * 24

class RelGTTokenizer(torch.nn.Module):
    """The multi-element tokenizer from the "Relational Graph Transformer"
    paper (https://arxiv.org/abs/2505.10960).

    For each of the B seed nodes v_i in a batch, the sampler provides K 
    sampled nodes v_j. Each v_j is turned into a token by embedding 
    five elements and combining their concatenation via learned matrix O:

        h_token(v_j) = O [h_feat(v_j) || h_type(v_j) || h_hop(v_i, v_j)
            || h_time(v_i, v_j) || h_PE(v_j)]

    where O is a learned (d x 5d) matrix, || denotes concatenation, and we 
    have the following elements:

    - h_feat encodes the row features of v_j via the given feature_encoder,
    - h_type embeds the node type (i.e., the table) of v_j,
    - h_hop embeds the hop distance between v_i and v_j,
    - h_time encodes the relative time tau(v_i) - tau(v_j) (converted from
      seconds to days) via PositionalEncoding followed by a linear
      projection, and
    - h_PE is a positional encoding computed on the sampled subgraphs (by
      default, SubgraphGNNPE).

    Hop distances 0, ..., max_hops denote sampled neighbors, while the extra
    hop value max_hops + 1 denotes randomly sampled filler nodes used when
    fewer than K neighbors exist.

    Args:
        channels (int): Size d of each token. Needs to be at least 4 if
            pe_encoder is None, since SubgraphGNNPE uses channels // 4
            hidden channels.
        node_types (List[str]): The node types (i.e., table names). The i-th
            entry is the node type with ID i in the node_type input of
            forward().
        feature_encoder (torch.nn.Module): A module mapping a dictionary of
            per-node-type inputs (e.g., torch_frame.TensorFrame objects) to a
            dictionary of node features of shape
            [num_nodes_of_type, channels], such as the HeteroEncoder of the
            RelBench pipeline.
        max_hops (int, optional): The maximum hop distance of sampled
            neighbors. 
        pe_encoder (torch.nn.Module, optional): The positional encoder,
            called as pe_encoder(edge_index, num_nodes, batch=batch,
            batch_size=batch_size) and returning a tensor of shape
            [num_nodes, channels]. If set to None, will use SubgraphGNNPE 
            with channels output channels.

    Shapes:
        - input:
          per-node-type inputs to feature_encoder,
          node indices (B, K),
          node type IDs (B, K),
          hop distances (B, K),
          relative times in seconds (B, K),
          subgraph edge indices (2, num_edges)
        - output: tokens (B, K, d)
    """
    def __init__(
        self,
        channels: int,
        node_types: List[str],
        feature_encoder: torch.nn.Module,
        max_hops: int = 2,
        pe_encoder: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        if len(node_types) == 0:
            raise ValueError("'node_types' needs to hold at least one entry")
        if len(set(node_types)) != len(node_types):
            raise ValueError(f"'node_types' must not hold duplicate entries "
                             f"(got {node_types})")
        if max_hops < 0:
            raise ValueError(f"'max_hops' must be non-negative "
                             f"(got {max_hops})")
        if pe_encoder is None and channels < 4:
            raise ValueError(f"'channels' must be at least 4 when using the "
                             f"default 'SubgraphGNNPE' (got {channels}); "
                             f"pass a custom 'pe_encoder' for smaller sizes")

        self.channels = channels
        self.node_types = list(node_types)
        self.max_hops = max_hops

        self.feature_encoder = feature_encoder
        self.type_emb = Embedding(len(node_types), channels)
        # Hops `0, ..., max_hops` plus one extra entry for filler nodes:
        self.hop_emb = Embedding(max_hops + 2, channels)
        # The sinusoidal encoding needs an even size (sine/cosine pairs), so
        # round up and let the linear projection map back to `channels`:
        time_channels = channels + channels % 2
        self.time_enc = PositionalEncoding(time_channels)
        self.time_lin = Linear(time_channels, channels)
        if pe_encoder is None:
            pe_encoder = SubgraphGNNPE(channels)
        self.pe_encoder = pe_encoder
        self.lin = Linear(5 * channels, channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        if hasattr(self.feature_encoder, 'reset_parameters'):
            self.feature_encoder.reset_parameters()
        self.type_emb.reset_parameters()
        self.hop_emb.reset_parameters()
        self.time_enc.reset_parameters()
        self.time_lin.reset_parameters()
        if hasattr(self.pe_encoder, 'reset_parameters'):
            self.pe_encoder.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        feat_dict: Dict[str, Any],
        node_idx: Tensor,
        node_type: Tensor,
        hop: Tensor,
        rel_time: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (Dict[str, Any]): The per-node-type inputs to
                feature_encoder, e.g., a dictionary holding a
                torch_frame.TensorFrame for each node type.
            node_idx (torch.Tensor): The index of each token within its node
                type, i.e., the row in the corresponding output of
                feature_encoder, of shape [B, K].
            node_type (torch.Tensor): The node type ID of each token, of shape
                [B, K].
            hop (torch.Tensor): The hop distance of each token from its seed
                node, of shape [B, K].
                Filler nodes are denoted by max_hops + 1.
            rel_time (torch.Tensor): The time of each seed node minus the time
                of each of its tokens in seconds, of shape [B, K].
            edge_index (torch.Tensor): The edge indices of the sampled
                subgraphs, where token k of seed node b is denoted by node
                index b * K + k.
        """
        if node_idx.dim() != 2:
            raise ValueError(f"Expected 'node_idx' to be two-dimensional "
                             f"(got {node_idx.dim()} dimensions)")
        for name, value in [('node_type', node_type), ('hop', hop),
                            ('rel_time', rel_time)]:
            if value.size() != node_idx.size():
                raise ValueError(f"Expected '{name}' to have shape "
                                 f"{tuple(node_idx.size())} "
                                 f"(got {tuple(value.size())})")

        B, K = node_idx.size()

        # Encode each table, then look up each node's row via (table, row)
        x_feat = self._gather_features(feat_dict, node_type, node_idx)

        x_type = self.type_emb(node_type)
        x_hop = self.hop_emb(hop)

        # Convert seconds to days
        days = rel_time.to(self.time_lin.weight.dtype) / SECONDS_PER_DAY
        # PositionalEncoding needs flat vector, so flatten to [B * K], 
        # encode, and reshape after
        x_time = self.time_lin(self.time_enc(days.view(-1))).view(B, K, -1)

        # One ID per-seed node, repeated K times 
        batch = torch.arange(B, device=node_idx.device).repeat_interleave(K)
        # Run positional encoder on all sampled subgraphs
        x_pe = self.pe_encoder(edge_index, B * K, batch=batch, batch_size=B)
        # All components need same shape
        x_pe = x_pe.view(B, K, -1)

        # Concatenate along the last dim to get one 5d-vector per token
        out = torch.cat([x_feat, x_type, x_hop, x_time, x_pe], dim=-1)
        # self.lin is the matrix O, mapping 5d back to d
        return self.lin(out)

    def _gather_features(
        self,
        feat_dict: Dict[str, Any],
        node_type: Tensor,
        node_idx: Tensor,
    ) -> Tensor:
        # Run feature encoder, returns dict of table name -> encoded rows
        x_dict = self.feature_encoder(feat_dict)

        out: Optional[Tensor] = None
        # i is the table ID, key is table name for x_dict
        for i, key in enumerate(self.node_types):
            # If we do not have a table, we should not have any nodes from
            # that table
            if key not in x_dict:
                if bool((node_type == i).any()):
                    raise ValueError(f"'feature_encoder' did not return "
                                     f"features for node type '{key}'")
                continue
            # Otherwise, feature encoder gives encoded table i
            x = x_dict[key]
            if x.dim() != 2 or x.size(-1) != self.channels:
                raise ValueError(f"Expected 'feature_encoder' to return "
                                 f"features of shape [*, {self.channels}] for "
                                 f"node type '{key}' "
                                 f"(got {tuple(x.size())})")
            if out is None:
                out = x.new_zeros(*node_idx.size(), self.channels)

            # True for nodes from table i
            mask = node_type == i
            # Get row numbers (in table i) for nodes of this node_type
            row_nums =  node_idx[mask]
            # Get rows from table i and copy them to the correct slots of 
            # output
            out[mask] = x[row_nums]

        if out is None:
            raise ValueError(f"'feature_encoder' did not return features for "
                             f"any node type in {self.node_types}")
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'num_node_types={len(self.node_types)}, '
                f'max_hops={self.max_hops})')
