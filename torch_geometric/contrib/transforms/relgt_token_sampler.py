from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce


@functional_transform('relgt_token_sampler')
class RelGTTokenSampler(BaseTransform):
    r"""The fixed-size token sampler from the `"Relational Graph Transformer"
    <https://arxiv.org/abs/2505.10960>`_ paper
    (functional name: :obj:`relgt_token_sampler`).

    Converts a heterogeneous mini-batch from
    :class:`~torch_geometric.loader.NeighborLoader` into :math:`K` tokens for
    each of its :math:`B` seed nodes :math:`v_i`, as expected by
    :class:`~torch_geometric.contrib.nn.encoding.RelGTTokenizer`.
    The mini-batch needs to be sampled in disjoint mode, *e.g.*, via temporal
    sampling.

    * The first token is the seed node :math:`v_i` itself.
    * If at least :math:`K - 1` nodes were sampled for :math:`v_i`, the
      remaining tokens are drawn from them without replacement.
    * If fewer nodes were sampled, each of them is used once and the remaining
      tokens are drawn from them with replacement.
    * If no nodes were sampled, the remaining tokens are drawn from all nodes
      of the mini-batch that do not lie in the future of :math:`v_i`, without
      replacement if there are enough of them.
      These filler tokens get the hop distance :obj:`max_hops + 1`.

    The last :math:`K - 1` tokens are shuffled.
    Edges between the tokens of a seed node are taken from the mini-batch and
    made undirected.
    Duplicated tokens share the edges of their node.

    .. code-block:: python

        from torch_geometric.contrib.transforms import RelGTTokenSampler
        from torch_geometric.loader import NeighborLoader

        loader = NeighborLoader(
            data,
            num_neighbors=[5000, 1000],
            input_nodes=('drivers', train_index),
            input_time=train_time,
            time_attr='time',
            batch_size=256,
            transform=RelGTTokenSampler(num_tokens=300),
        )

        for batch in loader:
            tokens = tokenizer(
                batch.tf_dict,
                batch.token_node_idx,
                batch.token_node_type,
                batch.token_hop,
                batch.token_rel_time,
                batch.token_edge_index,
            )

    The transform adds the following attributes to the mini-batch:

    * :obj:`token_node_idx` (:obj:`[B, K]`): The index of each token within
      its node type in the mini-batch.
    * :obj:`token_node_type` (:obj:`[B, K]`): The node type ID of each token,
      in the order of :obj:`data.node_types`.
    * :obj:`token_hop` (:obj:`[B, K]`): The hop distance of each token from
      its seed node. :obj:`max_hops` is the number of hops sampled by the
      loader.
    * :obj:`token_rel_time` (:obj:`[B, K]`): The time of each seed node minus
      the time of each of its tokens.
      This is :obj:`0` for the seed node itself, for node types without
      :obj:`time_attr`, and for all tokens if the mini-batch has no seed
      times.
    * :obj:`token_edge_index` (:obj:`[2, num_edges]`): The edges between
      tokens, where token :obj:`k` of seed node :obj:`b` has node index
      :obj:`b * K + k`.

    Args:
        num_tokens (int, optional): The number of tokens :math:`K` per seed
            node, including the seed node itself. (default: :obj:`300`)
        time_attr (str, optional): The name of the node-level attribute
            holding node timestamps. (default: :obj:`"time"`)
    """
    def __init__(self, num_tokens: int = 300, time_attr: str = 'time'):
        if num_tokens < 1:
            raise ValueError(f"'num_tokens' must be positive "
                             f"(got {num_tokens})")
        self.num_tokens = num_tokens
        self.time_attr = time_attr

    def forward(self, data: HeteroData) -> HeteroData:
        if not isinstance(data, HeteroData):
            raise ValueError(f"'{self.__class__.__name__}' expects a "
                             f"'HeteroData' object (got {type(data)})")

        input_type = self._get_input_type(data)
        num_seeds = int(data[input_type].batch_size)
        seed_time: Optional[Tensor] = getattr(data[input_type], 'seed_time',
                                              None)
        K = self.num_tokens

        offset, node_type, node_idx, seed, hop, time, has_time = \
            self._flatten_nodes(data, seed_time)
        max_hops = len(data[input_type].num_sampled_nodes) - 1
        seed_node = offset[input_type] + torch.arange(num_seeds,
                                                      device=node_idx.device)

        nbr, is_filler = self._sample_neighbors(seed, hop, num_seeds)
        empty = is_filler[:, 0] if K > 1 else is_filler.new_zeros(num_seeds)
        if bool(empty.any()):
            nbr[empty] = self._sample_fillers(
                seed_node[empty],
                seed_time[empty] if seed_time is not None else None, time,
                has_time)

        # Shuffle the order of the last `K - 1` tokens:
        perm = torch.rand(nbr.size(), device=nbr.device).argsort(dim=-1)
        nbr = nbr.gather(1, perm)
        is_filler = is_filler.gather(1, perm)

        token = torch.cat([seed_node.view(-1, 1), nbr], dim=1)  # [B, K]
        is_filler = torch.cat([is_filler.new_zeros(num_seeds, 1), is_filler],
                              dim=1)

        token_hop = hop[token]
        token_hop[is_filler] = max_hops + 1

        if seed_time is not None:
            token_rel_time = seed_time.view(-1, 1) - time[token]
            token_rel_time[~has_time[token]] = 0
            token_rel_time[:, 0] = 0
        else:
            token_rel_time = torch.zeros_like(token)

        data.token_node_idx = node_idx[token]
        data.token_node_type = node_type[token]
        data.token_hop = token_hop
        data.token_rel_time = token_rel_time
        data.token_edge_index = self._build_edge_index(
            data, offset, token, num_nodes=node_idx.numel())

        return data

    def _get_input_type(self, data: HeteroData) -> str:
        input_types = [
            key for key in data.node_types if 'input_id' in data[key]
        ]
        if len(input_types) != 1:
            raise ValueError(f"'{self.__class__.__name__}' expects a "
                             f"mini-batch from 'NeighborLoader' with exactly "
                             f"one input node type (got {input_types})")
        for key in data.node_types:
            if 'batch' not in data[key]:
                raise ValueError(f"'{self.__class__.__name__}' expects a "
                                 f"mini-batch from 'NeighborLoader' sampled "
                                 f"in disjoint mode (node type '{key}' has "
                                 f"no 'batch' vector). Pass 'disjoint=True' "
                                 f"or use temporal sampling.")
        return input_types[0]

    def _flatten_nodes(
        self,
        data: HeteroData,
        seed_time: Optional[Tensor],
    ) -> Tuple[Dict[str, int], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Concatenates the nodes of all node types into a single index
        space and returns their node type ID, index within their node type,
        seed node, hop distance, time and whether they have a time.
        """
        offset: Dict[str, int] = {}
        node_types: List[Tensor] = []
        node_idxs: List[Tensor] = []
        seeds: List[Tensor] = []
        hops: List[Tensor] = []
        times: List[Tensor] = []
        has_times: List[Tensor] = []

        time_dtype = seed_time.dtype if seed_time is not None else torch.long
        cumsum = 0
        for i, key in enumerate(data.node_types):
            store = data[key]
            batch = store.batch
            num_nodes = batch.numel()
            offset[key] = cumsum
            cumsum += num_nodes

            arange = torch.arange(num_nodes, device=batch.device)
            node_types.append(torch.full_like(arange, i))
            node_idxs.append(arange)
            seeds.append(batch)

            # Nodes are ordered by the hop in which they were sampled:
            num_sampled = torch.as_tensor(store.num_sampled_nodes,
                                          device=batch.device)
            hops.append(
                torch.bucketize(arange, num_sampled.cumsum(0), right=True))

            if self.time_attr in store:
                times.append(store[self.time_attr].to(time_dtype))
                has_times.append(torch.ones_like(arange, dtype=torch.bool))
            else:
                times.append(torch.zeros_like(arange, dtype=time_dtype))
                has_times.append(torch.zeros_like(arange, dtype=torch.bool))

        return (
            offset,
            torch.cat(node_types),
            torch.cat(node_idxs),
            torch.cat(seeds),
            torch.cat(hops),
            torch.cat(times),
            torch.cat(has_times),
        )

    def _sample_neighbors(
        self,
        seed: Tensor,
        hop: Tensor,
        num_seeds: int,
    ) -> Tuple[Tensor, Tensor]:
        r"""Samples :math:`K - 1` of the nodes sampled for each seed node.
        Returns the sampled nodes and a mask marking the slots of seed nodes
        without any sampled nodes, which need to be filled.
        """
        K = self.num_tokens
        device = seed.device

        # Group the candidate nodes by seed node in random order:
        cand = (hop > 0).nonzero().view(-1)
        cand = cand[torch.randperm(cand.numel(), device=device)]
        cand = cand[seed[cand].sort(stable=True)[1]]

        count = torch.bincount(seed[cand], minlength=num_seeds).view(-1, 1)
        ptr = count.cumsum(0) - count

        # Use each candidate once, then draw the remaining slots at random:
        slot = torch.arange(K - 1, device=device).view(1, -1)
        rand = torch.rand(num_seeds, K - 1, device=device) * count
        rand = rand.long().clamp(max=(count - 1).clamp(min=0))
        pos = torch.where(slot < count, slot, rand)

        is_filler = (count == 0).expand(-1, K - 1).clone()
        if cand.numel() == 0:
            return seed.new_zeros(num_seeds, K - 1), is_filler
        pos = (ptr + pos).clamp(max=cand.numel() - 1)
        return cand[pos], is_filler

    def _sample_fillers(
        self,
        seed_node: Tensor,
        seed_time: Optional[Tensor],
        time: Tensor,
        has_time: Tensor,
    ) -> Tensor:
        r"""Samples :math:`K - 1` random nodes from the whole mini-batch for
        each given seed node, excluding the seed node itself and nodes from
        its future. Samples without replacement if there are enough such
        nodes, and falls back to repeating the seed node if there are none.
        """
        K = self.num_tokens
        num_nodes = time.numel()
        device = time.device

        # Sort by time so that the valid fillers of each seed form a prefix.
        # Nodes without a time come first:
        key = time.to(torch.float64)
        key[~has_time] = float('-inf')
        key, order = key.sort()
        rank = torch.empty_like(order)
        rank[order] = torch.arange(num_nodes, device=device)

        if seed_time is not None:
            num_valid = torch.searchsorted(key, seed_time.to(torch.float64),
                                           right=True)
        else:
            num_valid = torch.full_like(seed_node, num_nodes)

        # Skip the seed node if it lies in the prefix:
        seed_rank = rank[seed_node]
        seed_valid = seed_rank < num_valid
        num_avail = (num_valid - seed_valid.long()).view(-1, 1)

        rand = torch.rand(seed_node.numel(), K - 1, device=device) * num_avail
        rand = rand.long().clamp(max=(num_avail - 1).clamp(min=0))

        # Sample without replacement by keeping the nodes with the `K - 1`
        # largest random numbers:
        enough = (num_avail >= K - 1).view(-1).nonzero().view(-1)
        if enough.numel() > 0:
            max_avail = int(num_avail[enough].max())
            arange = torch.arange(max_avail, device=device).view(1, -1)
            for chunk in enough.split(max(1, 2**24 // max_avail)):
                keys = torch.rand(chunk.numel(), max_avail, device=device)
                keys[arange >= num_avail[chunk]] = -1.0
                rand[chunk] = keys.topk(K - 1, dim=1).indices

        rand += (seed_valid.view(-1, 1) & (rand >= seed_rank.view(-1, 1)))

        out = order[rand.clamp(max=num_nodes - 1)]
        return torch.where(num_avail > 0, out, seed_node.view(-1, 1))

    def _build_edge_index(
        self,
        data: HeteroData,
        offset: Dict[str, int],
        token: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Maps the sampled edges of the mini-batch to edges between the
        tokens of each seed node.
        """
        K = self.num_tokens
        device = token.device

        rows: List[Tensor] = []
        cols: List[Tensor] = []
        for edge_type in data.edge_types:
            store = data[edge_type]
            if 'edge_index' not in store:
                continue
            src, _, dst = edge_type
            rows.append(store.edge_index[0] + offset[src])
            cols.append(store.edge_index[1] + offset[dst])

        if len(rows) == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        # Treat edges as undirected:
        u = torch.cat(rows + cols)
        v = torch.cat(cols + rows)

        # Key token (row `b`, node `n`) as `b * num_nodes + n`, and map
        # duplicated tokens to the first token of their node:
        token = token.view(-1)
        token_id = torch.arange(token.numel(), device=device)
        token_row = token_id.div(K, rounding_mode='floor')
        token_key = token_row * num_nodes + token
        uniq_key, inv = token_key.unique(return_inverse=True)
        first = torch.full((uniq_key.numel(), ), token.numel(), device=device)
        first = first.scatter_reduce(0, inv, token_id, reduce='amin')

        # Expand every edge `(u, v)` to all tokens of `u`:
        node_order = token.argsort(stable=True)
        node_count = torch.bincount(token, minlength=num_nodes)
        node_ptr = node_count.cumsum(0) - node_count

        num_expand = node_count[u]
        edge = torch.arange(u.numel(), device=device)
        edge = edge.repeat_interleave(num_expand)
        expand_ptr = num_expand.cumsum(0) - num_expand
        within = torch.arange(edge.numel(), device=device)
        within -= expand_ptr.repeat_interleave(num_expand)
        src = node_order[node_ptr[u[edge]] + within]

        # Connect them to the first token of `v` in the same row, if any:
        dst_key = token_row[src] * num_nodes + v[edge]
        idx = torch.searchsorted(uniq_key, dst_key)
        idx = idx.clamp(max=uniq_key.numel() - 1)
        mask = uniq_key[idx] == dst_key
        src, dst = src[mask], first[idx[mask]]

        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack(
            [torch.cat([src, dst]),
             torch.cat([dst, src])])
        return coalesce(edge_index, num_nodes=token.numel())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tokens={self.num_tokens})'
