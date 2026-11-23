import pytest
import torch

from torch_geometric.contrib.transforms import RelGTTokenSampler
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.testing import onlyNeighborSampler


def get_batch(with_time: bool = True) -> HeteroData:
    r"""Returns a hand-crafted disjoint mini-batch.

    The mini-batch holds three seed nodes of type :obj:`'u'` sampled over two
    hops:

    * seed 0: 1-hop neighbors :obj:`o0, o1` and 2-hop neighbor :obj:`u3`
    * seed 1: 1-hop neighbor :obj:`o2` and 2-hop neighbor :obj:`p0`
    * seed 2: no neighbors

    Node type :obj:`'p'` carries no timestamps.
    """
    data = HeteroData()
    data['u'].batch = torch.tensor([0, 1, 2, 0])
    data['u'].num_sampled_nodes = [3, 0, 1]
    data['o'].batch = torch.tensor([0, 0, 1])
    data['o'].num_sampled_nodes = [0, 3, 0]
    data['p'].batch = torch.tensor([1])
    data['p'].num_sampled_nodes = [0, 0, 1]
    data['p'].num_nodes = 1

    data['o', 'rev', 'u'].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    data['u', 'to', 'o'].edge_index = torch.tensor([[3], [0]])
    data['p', 'in', 'o'].edge_index = torch.tensor([[0], [2]])

    data['u'].input_id = torch.arange(3)
    data['u'].batch_size = 3
    if with_time:
        data['u'].time = torch.tensor([10, 20, 7, 5])
        data['o'].time = torch.tensor([8, 9, 15])
        data['u'].seed_time = torch.tensor([10, 20, 7])
    return data


# The node types in `get_batch()` and the (node type, index) of every node:
U, O, P = 0, 1, 2
U0, U1, U2, U3 = (U, 0), (U, 1), (U, 2), (U, 3)
O0, O1, O2 = (O, 0), (O, 1), (O, 2)
P0 = (P, 0)


def tokens_of(data: HeteroData, b: int):
    return list(
        zip(data.token_node_type[b].tolist(), data.token_node_idx[b].tolist()))


def test_relgt_token_sampler_shapes():
    transform = RelGTTokenSampler(num_tokens=6)
    assert str(transform) == 'RelGTTokenSampler(num_tokens=6)'

    data = transform(get_batch())
    for key in [
            'token_node_idx', 'token_node_type', 'token_hop', 'token_rel_time'
    ]:
        assert data[key].size() == (3, 6)
        assert data[key].dtype == torch.long
    assert data.token_edge_index.size(0) == 2
    assert data.token_edge_index.dtype == torch.long


def test_relgt_token_sampler_seed_first():
    data = RelGTTokenSampler(num_tokens=6)(get_batch())
    assert data.token_node_type[:, 0].tolist() == [U, U, U]
    assert data.token_node_idx[:, 0].tolist() == [0, 1, 2]
    assert data.token_hop[:, 0].tolist() == [0, 0, 0]
    assert data.token_rel_time[:, 0].tolist() == [0, 0, 0]


def test_relgt_token_sampler_resampling():
    data = RelGTTokenSampler(num_tokens=6)(get_batch())

    # Seeds with fewer than `K - 1` neighbors use each of them at least once
    # and repeat only their own neighbors:
    assert set(tokens_of(data, 0)[1:]) == {O0, O1, U3}
    assert set(tokens_of(data, 1)[1:]) == {O2, P0}

    # Duplicated tokens keep their actual hop distance:
    hop = {O0: 1, O1: 1, U3: 2, O2: 1, P0: 2}
    for b in [0, 1]:
        for k, token in enumerate(tokens_of(data, b)[1:], start=1):
            assert data.token_hop[b, k] == hop[token]


def test_relgt_token_sampler_without_replacement():
    data = RelGTTokenSampler(num_tokens=3)(get_batch())

    # Seed 0 has three neighbors to choose `K - 1 = 2` from:
    tokens = tokens_of(data, 0)[1:]
    assert len(set(tokens)) == 2
    assert set(tokens) <= {O0, O1, U3}


def test_relgt_token_sampler_fillers():
    torch.manual_seed(12345)
    data = RelGTTokenSampler(num_tokens=50)(get_batch())

    # Seed 2 has no neighbors, so its tokens are random nodes of the batch
    # that are not in its future (`u3` at time 5 and `p0` without time):
    assert set(tokens_of(data, 2)[1:]) == {U3, P0}
    assert data.token_hop[2, 1:].tolist() == [3] * 49

    # Seeds with neighbors never receive filler nodes:
    assert int((data.token_hop[:2] == 3).sum()) == 0


def test_relgt_token_sampler_filler_fallback():
    data = get_batch()
    # No node lies in the past of seed 2 anymore:
    data['u'].time[3] = 100
    data['p'].time = torch.tensor([100])
    data = RelGTTokenSampler(num_tokens=4)(data)

    # Seed 2 falls back to repeating itself:
    assert tokens_of(data, 2) == [U2] * 4
    assert data.token_hop[2].tolist() == [0, 3, 3, 3]
    assert data.token_rel_time[2].tolist() == [0, 0, 0, 0]


def test_relgt_token_sampler_rel_time():
    data = RelGTTokenSampler(num_tokens=6)(get_batch())

    time = {O0: 8, O1: 9, U3: 5, O2: 15}
    seed_time = [10, 20, 7]
    for b in range(3):
        for k, token in enumerate(tokens_of(data, b)[1:], start=1):
            # Node type 'p' has no time, so its relative time is zero:
            expected = seed_time[b] - time[token] if token != P0 else 0
            assert data.token_rel_time[b, k] == expected
    assert int(data.token_rel_time.min()) >= 0


def test_relgt_token_sampler_without_time():
    data = RelGTTokenSampler(num_tokens=6)(get_batch(with_time=False))
    assert data.token_rel_time.abs().sum() == 0

    # Without time, all other nodes of the batch are valid fillers:
    assert set(tokens_of(data, 2)[1:]) <= {U0, U1, U3, O0, O1, O2, P0}


def test_relgt_token_sampler_edge_index():
    K = 6
    data = RelGTTokenSampler(num_tokens=K)(get_batch())
    edge_index = data.token_edge_index
    tokens = [tokens_of(data, b) for b in range(3)]

    # The (undirected) edges of the mini-batch:
    edges = {(O0, U0), (O1, U0), (O2, U1), (U3, O0), (P0, O2)}
    edges |= {(v, u) for u, v in edges}

    # Edges are undirected, never cross between seeds, and exist in the batch:
    pairs = set(map(tuple, edge_index.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    for i, j in pairs:
        assert i // K == j // K
        b = i // K
        assert (tokens[b][i % K], tokens[b][j % K]) in edges

    # Every copy of a duplicated token is connected to its neighbors:
    for b in range(2):
        for k, token in enumerate(tokens[b]):
            neighbors = {tokens[b][j % K] for i, j in pairs if i == b * K + k}
            expected = {v for u, v in edges if u == token}
            assert neighbors == expected & set(tokens[b])

    # Seed 2 only holds unconnected filler nodes:
    assert all(i // K != 2 for i, _ in pairs)


def test_relgt_token_sampler_single_token():
    data = RelGTTokenSampler(num_tokens=1)(get_batch())
    assert data.token_node_idx.tolist() == [[0], [1], [2]]
    assert data.token_edge_index.numel() == 0


def test_relgt_token_sampler_invalid_inputs():
    with pytest.raises(ValueError, match="must be positive"):
        RelGTTokenSampler(num_tokens=0)

    transform = RelGTTokenSampler(num_tokens=6)
    with pytest.raises(ValueError, match="expects a 'HeteroData'"):
        transform(Data())

    data = get_batch()
    del data['o'].batch
    with pytest.raises(ValueError, match="disjoint mode"):
        transform(data)

    data = get_batch()
    del data['u'].input_id
    with pytest.raises(ValueError, match="exactly one input node type"):
        transform(data)


@onlyNeighborSampler
@pytest.mark.parametrize('num_neighbors', [[5], [5, 3]])
def test_relgt_token_sampler_neighbor_loader(num_neighbors):
    torch.manual_seed(12345)
    data = HeteroData()
    data['u'].x = torch.randn(20, 4)
    data['u'].time = torch.randint(0, 100, (20, ))
    data['o'].x = torch.randn(50, 4)
    data['o'].time = torch.randint(0, 100, (50, ))
    src = torch.randint(0, 20, (50, ))
    dst = torch.arange(50)
    data['u', 'to', 'o'].edge_index = torch.stack([src, dst])
    data['o', 'rev', 'u'].edge_index = torch.stack([dst, src])

    K = 8
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes='u',
        input_time=data['u'].time,
        time_attr='time',
        batch_size=10,
        transform=RelGTTokenSampler(num_tokens=K),
    )
    max_hops = len(num_neighbors)
    node_types = ['u', 'o']
    for batch in loader:
        B = batch['u'].batch_size
        assert batch.token_node_idx.size() == (B, K)
        assert int(batch.token_hop.max()) <= max_hops + 1

        # No token lies in the future of its seed node:
        assert int(batch.token_rel_time.min()) >= 0
        for b in range(B):
            for k in range(K):
                key = node_types[int(batch.token_node_type[b, k])]
                global_id = batch[key].n_id[batch.token_node_idx[b, k]]
                time = data[key].time[global_id]
                assert time <= batch['u'].seed_time[b]
                if k > 0:
                    rel_time = batch['u'].seed_time[b] - time
                    assert batch.token_rel_time[b, k] == rel_time

        # Every token edge corresponds to an edge in the full graph:
        edges = set(zip(src.tolist(), dst.tolist()))
        for i, j in batch.token_edge_index.t().tolist():
            assert i // K == j // K
            nodes = []
            for t in [i, j]:
                b, k = t // K, t % K
                key = node_types[int(batch.token_node_type[b, k])]
                local = batch.token_node_idx[b, k]
                nodes.append((key, int(batch[key].n_id[local])))
            (key_i, i_id), (key_j, j_id) = nodes
            assert key_i != key_j
            pair = (i_id, j_id) if key_i == 'u' else (j_id, i_id)
            assert pair in edges
