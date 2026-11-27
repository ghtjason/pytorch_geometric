import pytest
import torch

from torch_geometric.contrib.nn.encoding import SubgraphGNNPE
from torch_geometric.contrib.nn.models import RelGT
from torch_geometric.contrib.transforms import RelGTTokenSampler
from torch_geometric.data import HeteroData

NODE_TYPES = ['user', 'item']


class DictEncoder(torch.nn.Module):
    # Stand-in for a per-table encoder (e.g., RelBench's `HeteroEncoder`):
    def __init__(self, in_channels, channels, node_types=NODE_TYPES):
        super().__init__()
        self.lins = torch.nn.ModuleDict({
            key:
            torch.nn.Linear(in_channels, channels)
            for key in node_types
        })

    def forward(self, feat_dict):
        return {key: self.lins[key](x) for key, x in feat_dict.items()}


class FixedFeaturePE(torch.nn.Module):
    # Runs `SubgraphGNNPE` on fixed instead of random node features:
    def __init__(self, pe, x):
        super().__init__()
        self.pe = pe
        self.x = x

    def forward(self, edge_index, num_nodes, batch=None, batch_size=None):
        return self.pe(edge_index, num_nodes, x=self.x, batch=batch,
                       batch_size=batch_size)


def get_inputs():
    # Three seed nodes with five tokens each:
    feat_dict = {
        'user': torch.randn(4, 4),
        'item': torch.randn(6, 4),
    }
    node_idx = torch.tensor([[0, 2, 4, 1, 5], [1, 0, 3, 3, 2], [2, 5, 0, 3,
                                                                1]])
    node_type = torch.tensor([[0, 1, 1, 0, 1], [0, 1, 1, 1, 0],
                              [0, 1, 0, 1, 1]])
    hop = torch.tensor([[0, 1, 2, 1, 2], [0, 1, 1, 2, 3], [0, 1, 2, 2, 1]])
    rel_time = torch.tensor([[0., 3600., 86400., 1e7, 60.],
                             [0., 0., 60., 1e5, 1e3], [0., 1e4, 2e4, 3e4,
                                                       4e4]])
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 3, 5, 6, 6, 7, 10, 11, 10, 14],
        [1, 0, 2, 1, 3, 0, 6, 5, 7, 6, 11, 10, 14, 10],
    ])
    return feat_dict, node_idx, node_type, hop, rel_time, edge_index


@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('heads', [1, 2])
def test_relgt(num_layers, heads):
    model = RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), num_layers=num_layers,
                  heads=heads, num_centroids=4)
    assert str(model) == (f'RelGT(8, 3, num_layers={num_layers}, '
                          f'heads={heads}, global_attn=True)')
    assert len(model.layers) == num_layers

    out = model(*get_inputs())
    assert out.size() == (3, 3)

    model.eval()
    assert model(*get_inputs()).size() == (3, 3)


def test_relgt_backward():
    model = RelGT(8, 1, NODE_TYPES, DictEncoder(4, 8), num_centroids=4)
    model(*get_inputs()).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name


def test_relgt_neighbor_permutation_invariance():
    torch.manual_seed(12345)
    B, K = 3, 5
    feat_dict, node_idx, node_type, hop, rel_time, edge_index = get_inputs()

    x = torch.randn(B * K, 1)
    pe = FixedFeaturePE(SubgraphGNNPE(8, num_layers=2), x)
    model = RelGT(8, 2, NODE_TYPES, DictEncoder(4, 8), heads=2,
                  num_centroids=4, pe_encoder=pe).eval()
    out = model(feat_dict, node_idx, node_type, hop, rel_time, edge_index)

    # Reorder the neighbor tokens (but not the seed token) of every seed node,
    # together with their random features and subgraph edges:
    perm = torch.stack([
        torch.cat(
            [torch.zeros(1, dtype=torch.long),
             torch.randperm(K - 1) + 1]) for _ in range(B)
    ])
    flat_perm = (perm + K * torch.arange(B).view(-1, 1)).view(-1)
    inv_perm = torch.empty_like(flat_perm)
    inv_perm[flat_perm] = torch.arange(B * K)
    assert not torch.equal(flat_perm, torch.arange(B * K))

    pe.x = x[flat_perm]
    out_perm = model(feat_dict, node_idx.gather(1, perm),
                     node_type.gather(1, perm), hop.gather(1, perm),
                     rel_time.gather(1, perm), inv_perm[edge_index])
    assert torch.allclose(out, out_perm, atol=1e-5)


def test_relgt_without_global_attn():
    model = RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), global_attn=False)
    assert str(model).endswith('global_attn=False)')
    assert model.global_attn is None
    assert model.ffn[0].in_channels == 8
    assert model(*get_inputs()).size() == (3, 3)


@pytest.mark.parametrize('norm', ['layer_norm', None])
def test_relgt_norm(norm):
    model = RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), num_centroids=4,
                  act='relu', norm=norm)
    out = model(*get_inputs())
    assert out.size() == (3, 3)
    out.sum().backward()


def test_relgt_global_channels():
    model = RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), num_centroids=4)
    assert model.global_attn.in_channels == 4
    assert model.global_attn.centroids.size() == (4, 4)

    model = RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), num_centroids=4,
                  global_channels=6)
    assert model.global_attn.in_channels == 6
    assert model(*get_inputs()).size() == (3, 3)


def test_relgt_centroid_updates():
    model = RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), num_centroids=4)

    centroids = model.global_attn.centroids.clone()
    model(*get_inputs())
    assert not torch.equal(model.global_attn.centroids, centroids)

    model.eval()
    centroids = model.global_attn.centroids.clone()
    model(*get_inputs())
    assert torch.equal(model.global_attn.centroids, centroids)


def test_relgt_with_token_sampler():
    # A disjoint `NeighborLoader` mini-batch with three 'user' seed nodes:
    data = HeteroData()
    data['user'].batch = torch.tensor([0, 1, 2, 0])
    data['user'].num_sampled_nodes = [3, 0, 1]
    data['user'].time = torch.tensor([10, 20, 7, 5])
    data['item'].batch = torch.tensor([0, 0, 1])
    data['item'].num_sampled_nodes = [0, 3, 0]
    data['item'].time = torch.tensor([8, 9, 15])
    data['item', 'rev', 'user'].edge_index = torch.tensor([[0, 1, 2],
                                                           [0, 0, 1]])
    data['user', 'to', 'item'].edge_index = torch.tensor([[3], [0]])
    data['user'].input_id = torch.arange(3)
    data['user'].batch_size = 3
    data['user'].seed_time = torch.tensor([10, 20, 7])
    data = RelGTTokenSampler(num_tokens=6)(data)

    feat_dict = {'user': torch.randn(4, 4), 'item': torch.randn(3, 4)}
    model = RelGT(8, 1, data.node_types, DictEncoder(4, 8, data.node_types),
                  num_centroids=4)
    out = model(feat_dict, data.token_node_idx, data.token_node_type,
                data.token_hop, data.token_rel_time, data.token_edge_index)
    assert out.size() == (3, 1)


def test_relgt_reset_parameters():
    model = RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), num_centroids=4)
    model(*get_inputs())
    model.reset_parameters()
    assert torch.equal(model.global_attn.cluster_size, torch.ones(4))
    assert int(model.ffn_in_norm.module.num_batches_tracked) == 0
    assert int(model.layers[0].ffn_in_norm.module.num_batches_tracked) == 0


def test_relgt_invalid_inputs():
    with pytest.raises(ValueError, match="'num_layers' must be positive"):
        RelGT(8, 3, NODE_TYPES, DictEncoder(4, 8), num_layers=0)
