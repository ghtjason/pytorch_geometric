import inspect
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, MultiheadAttention, Sequential

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


class RelGTLayer(torch.nn.Module):
    r"""The local transformer layer from the `"Relational Graph Transformer"
    <https://arxiv.org/abs/2505.10960>`_ paper.

    The :math:`K` tokens of each seed node attend to each other via
    multi-head self-attention, followed by a feed-forward network.
    Both blocks use pre-normalized residual connections:

    .. math::
        \mathbf{X} &\leftarrow \mathbf{X} + \mathrm{Dropout} \left(
        \mathrm{Attention} \left( \mathrm{LayerNorm}(\mathbf{X}) \right)
        \right)

        \mathbf{X} &\leftarrow \mathbf{X} + \mathrm{FFN} \left(
        \mathrm{LayerNorm}(\mathbf{X}) \right)

    where :math:`\mathrm{FFN}` is a two-layer MLP with activation and dropout
    between two normalization layers.
    By default, these apply batch normalization over all tokens of the
    mini-batch.
    For the global module of RelGT, see
    :class:`~torch_geometric.contrib.nn.attention.GlobalCentroidAttention`.

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
        act (str or Callable, optional): The non-linear activation function to
            use in the feed-forward network. (default: :obj:`"gelu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str, optional): The normalization function to use before and
            after the feed-forward network.
            Graph-level normalizations treat the tokens of each seed node as a
            separate graph. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
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
        act: Union[str, Callable, None] = 'gelu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
        self.ffn = Sequential(
            Linear(channels, self.ffn_channels),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(self.ffn_channels, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.ffn_in_norm = normalization_resolver(norm, channels,
                                                  **norm_kwargs)
        self.ffn_out_norm = normalization_resolver(norm, channels,
                                                   **norm_kwargs)

        self.norm_with_batch = False
        if self.ffn_in_norm is not None:
            signature = inspect.signature(self.ffn_in_norm.forward)
            self.norm_with_batch = 'batch' in signature.parameters

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.attn_norm.reset_parameters()
        self.attn._reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn[0].reset_parameters()
        self.ffn[3].reset_parameters()
        if self.ffn_in_norm is not None:
            self.ffn_in_norm.reset_parameters()
        if self.ffn_out_norm is not None:
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

        # Normalization is applied over all `B * K` tokens:
        B, K = x.size(0), x.size(1)
        h = self.ffn_norm(x).view(-1, self.channels)
        h = self.ffn(self._norm(self.ffn_in_norm, h, B, K))
        h = self._norm(self.ffn_out_norm, h, B, K)
        return x + h.view(x.size())

    def _norm(
        self,
        norm: Optional[torch.nn.Module],
        x: Tensor,
        B: int,
        K: int,
    ) -> Tensor:
        if norm is None:
            return x
        if self.norm_with_batch:
            batch = torch.arange(B, device=x.device).repeat_interleave(K)
            return norm(x, batch, batch_size=B)
        return norm(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'heads={self.heads})')
