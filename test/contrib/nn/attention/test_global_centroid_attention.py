import pytest
import torch

from torch_geometric.contrib.nn.attention import GlobalCentroidAttention
from torch_geometric.testing import withDevice


def clone_buffers(module):
    return {key: value.clone() for key, value in module.named_buffers()}


@withDevice
@pytest.mark.parametrize('heads', [1, 2])
def test_global_centroid_attention(device, heads):
    x = torch.randn(10, 16, device=device)
    model = GlobalCentroidAttention(16, 32, num_centroids=8, heads=heads)
    model = model.to(device)
    assert str(model) == (f'GlobalCentroidAttention(16, 32, num_centroids=8, '
                          f'heads={heads})')

    out = model(x)
    assert out.size() == (10, 32)
    assert out.device == x.device

    model.eval()
    assert torch.allclose(model(x), model(x))


def test_global_centroid_attention_backward():
    x = torch.randn(10, 16, requires_grad=True)
    model = GlobalCentroidAttention(16, 16, num_centroids=8, heads=2)
    model(x).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, param in model.named_parameters():
        assert param.grad is not None, name
    for buffer in model.buffers():
        assert not buffer.requires_grad


def test_global_centroid_attention_eval_keeps_centroids():
    model = GlobalCentroidAttention(16, 16, num_centroids=8).eval()
    before = clone_buffers(model)
    model(torch.randn(10, 16))
    for key, value in model.named_buffers():
        assert torch.equal(value, before[key]), key


def test_global_centroid_attention_kmeans_step():
    torch.manual_seed(12345)
    num_centroids, decay, eps = 4, 0.9, 1e-5
    model = GlobalCentroidAttention(3, 3, num_centroids, decay=decay)
    x = torch.randn(20, 3)

    centroids = model.centroids.clone()
    size = model.cluster_size.clone()
    total = model.cluster_sum.clone()

    model(x)

    # Reference EMA k-means step, computed one query at a time:
    x = (x - x.mean(dim=0)) / (x.var(dim=0, unbiased=False) + eps).sqrt()
    count = torch.zeros(num_centroids)
    assigned_sum = torch.zeros(num_centroids, 3)
    for i in range(x.size(0)):
        b = int(((x[i] - centroids)**2).sum(dim=-1).argmin())
        count[b] += 1
        assigned_sum[b] += x[i]

    size = decay * size + (1 - decay) * count
    n = size.sum()
    size = (size + eps) / (n + num_centroids * eps) * n
    total = decay * total + (1 - decay) * assigned_sum

    assert torch.allclose(model.cluster_size, size, atol=1e-6)
    assert torch.allclose(model.cluster_sum, total, atol=1e-6)
    assert torch.allclose(model.centroids, total / size.view(-1, 1), atol=1e-6)


def test_global_centroid_attention_count_bias():
    model = GlobalCentroidAttention(4, 4, num_centroids=3).eval()
    with torch.no_grad():
        model.lin_key.weight.zero_()
        model.lin_key.bias.zero_()
        model.cluster_size.copy_(torch.tensor([1.0, 2.0, 5.0]))

    # With equal logits, attention weights are proportional to the sizes:
    std = (model.norm.running_var + model.eps).sqrt()
    centroids = model.centroids * std + model.norm.running_mean
    weight = torch.tensor([1.0, 2.0, 5.0]) / 8.0
    expected = weight @ model.lin_value(centroids)

    out = model(torch.randn(6, 4))
    assert torch.allclose(out, expected.expand(6, -1), atol=1e-6)


def test_global_centroid_attention_scale():
    model = GlobalCentroidAttention(4, 8, num_centroids=3, heads=2).eval()
    x = torch.randn(5, 4)

    # Logits are scaled by the size of each head:
    std = (model.norm.running_var + model.eps).sqrt()
    centroids = model.centroids * std + model.norm.running_mean
    q = model.lin_query(x).view(5, 2, 4)
    k = model.lin_key(centroids).view(3, 2, 4)
    v = model.lin_value(centroids).view(3, 2, 4)
    alpha = (torch.einsum('nhc,bhc->nhb', q, k) / 4**0.5).softmax(dim=-1)
    expected = torch.einsum('nhb,bhc->nhc', alpha, v).reshape(5, 8)
    assert torch.allclose(model(x), expected, atol=1e-6)


def test_global_centroid_attention_single_query():
    model = GlobalCentroidAttention(4, 4, num_centroids=3)
    before = clone_buffers(model)
    assert model(torch.randn(1, 4)).size() == (1, 4)
    assert torch.equal(model.centroids, before['centroids'])


def test_global_centroid_attention_reset_parameters():
    model = GlobalCentroidAttention(4, 4, num_centroids=3)
    model(torch.randn(10, 4))
    model.reset_parameters()
    assert torch.equal(model.cluster_size, torch.ones(3))
    assert torch.equal(model.cluster_sum, model.centroids)
    assert torch.equal(model.norm.running_mean, torch.zeros(4))
    assert int(model.norm.num_batches_tracked) == 0


def test_global_centroid_attention_invalid_inputs():
    with pytest.raises(ValueError, match="must be divisible by 'heads'"):
        GlobalCentroidAttention(8, 10, heads=3)
    with pytest.raises(ValueError, match="'num_centroids' must be positive"):
        GlobalCentroidAttention(8, 8, num_centroids=0)
    with pytest.raises(ValueError, match="'decay' must be in"):
        GlobalCentroidAttention(8, 8, decay=1.0)

    model = GlobalCentroidAttention(8, 8, num_centroids=4)
    with pytest.raises(ValueError, match="Expected 'x' to have shape"):
        model(torch.randn(5, 4))
