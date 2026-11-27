import pytest
import torch

from torch_geometric.contrib.nn.attention import RelGTLayer
from torch_geometric.testing import withDevice


@withDevice
@pytest.mark.parametrize('heads', [1, 2])
def test_relgt_layer(device, heads):
    x = torch.randn(3, 5, 16, device=device)
    model = RelGTLayer(16, heads=heads).to(device)
    assert str(model) == f'RelGTLayer(16, heads={heads})'
    assert model.ffn_channels == 32

    out = model(x)
    assert out.size() == (3, 5, 16)
    assert out.device == x.device


def test_relgt_layer_backward():
    x = torch.randn(3, 5, 16, requires_grad=True)
    model = RelGTLayer(16, heads=2, ffn_channels=8)
    model(x).sum().backward()

    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, name


def test_relgt_layer_permutation_equivariance():
    torch.manual_seed(12345)
    x = torch.randn(3, 6, 16)
    model = RelGTLayer(16, heads=2).eval()

    # Reordering the tokens of a seed node reorders its outputs alike:
    perm = torch.randperm(6)
    out = model(x)
    assert torch.allclose(model(x[:, perm]), out[:, perm], atol=1e-5)


def test_relgt_layer_seed_independence():
    torch.manual_seed(12345)
    x = torch.randn(3, 6, 16)
    model = RelGTLayer(16, heads=2).eval()

    # Tokens only attend to tokens of the same seed node:
    out1 = model(x)
    x[1] = torch.randn(6, 16)
    out2 = model(x)
    assert torch.allclose(out1[0], out2[0], atol=1e-6)
    assert torch.allclose(out1[2], out2[2], atol=1e-6)
    assert not torch.allclose(out1[1], out2[1], atol=1e-6)


@pytest.mark.parametrize('norm', ['layer_norm', None])
def test_relgt_layer_norm(norm):
    torch.manual_seed(12345)
    x = torch.randn(3, 6, 16)
    model = RelGTLayer(16, heads=2, act='relu', norm=norm)

    # Unlike batch normalization, seed nodes stay independent in training:
    out1 = model(x)
    x[1] = torch.randn(6, 16)
    out2 = model(x)
    assert torch.allclose(out1[0], out2[0], atol=1e-6)
    assert torch.allclose(out1[2], out2[2], atol=1e-6)


def test_relgt_layer_residual():
    x = torch.randn(3, 6, 16)
    model = RelGTLayer(16, heads=2).eval()

    # Without attention and feed-forward outputs, the layer is the identity:
    with torch.no_grad():
        model.attn.out_proj.weight.zero_()
        model.attn.out_proj.bias.zero_()
        model.ffn[3].weight.zero_()
        model.ffn[3].bias.zero_()
    assert torch.allclose(model(x), x, atol=1e-6)


def test_relgt_layer_reset_parameters():
    model = RelGTLayer(16, heads=2)
    model(torch.randn(3, 6, 16))
    with torch.no_grad():
        model.attn.out_proj.bias.fill_(1.0)
    model.reset_parameters()
    assert torch.equal(model.attn.out_proj.bias, torch.zeros(16))
    assert int(model.ffn_in_norm.module.num_batches_tracked) == 0
    assert int(model.ffn_out_norm.module.num_batches_tracked) == 0


def test_relgt_layer_invalid_inputs():
    with pytest.raises(ValueError, match="must be divisible by 'heads'"):
        RelGTLayer(10, heads=3)

    model = RelGTLayer(8)
    with pytest.raises(ValueError, match="Expected 'x' to have shape"):
        model(torch.randn(5, 8))
    with pytest.raises(ValueError, match="Expected 'x' to have shape"):
        model(torch.randn(2, 5, 4))
