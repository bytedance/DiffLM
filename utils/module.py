# Copyright (c) 2023 amazon-science
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/amazon-science/tabsyn/blob/main/tabsyn/model.py.
#
# This modified file is released under the same license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Union


ModuleType = Union[str, Callable[..., nn.Module]]


def split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    # bsz, q_len, _ = tensor.size()
    # return tensor.view(bsz, q_len, num_heads, attn_head_size).transpose(1, 2)
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    return tensor.view(new_shape).permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


def safe_log(z):
    return torch.log(z + 1e-7)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def generate_grid(zmin, zmax, dz, device, ndim=2):
    """generate a 1- or 2-dimensional grid
    Returns: Tensor, int
        Tensor: The grid tensor with shape (k^2, 2),
            where k=(zmax - zmin)/dz
        int: k
    """

    if ndim == 2:
        x = torch.arange(zmin, zmax, dz)
        k = x.size(0)

        x1 = x.unsqueeze(1).repeat(1, k).view(-1)
        x2 = x.repeat(k)

        return torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1).to(device), k

    elif ndim == 1:
        return torch.arange(zmin, zmax, dz).unsqueeze(1).to(device)


class CausalLMPooling(nn.Module):
    def __init__(self, strategy="mean"):
        super(CausalLMPooling, self).__init__()
        self.strategy = strategy

    def forward(self, last_hidden_state, attention_mask):
        if self.strategy == "mean":
            pooled = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
            pooled.masked_fill_(torch.isnan(pooled), 0)
        elif self.strategy == "last":
            def _extract_last_nonzero(m):
                nonzeros = (m == 1).nonzero(as_tuple=True)[0]
                return torch.max(nonzeros) if nonzeros.size(0) > 0 else 0
            last_indices = torch.tensor([_extract_last_nonzero(m) for m in attention_mask])
            i = torch.arange(last_hidden_state.shape[0]).reshape(last_hidden_state.shape[0], 1, 1)
            j = last_indices.reshape(last_indices.shape[0], 1, 1)
            k = torch.arange(last_hidden_state.shape[2])
            pooled = last_hidden_state[i, j, k][:, 0, :]
            pooled.masked_fill_(torch.isnan(pooled), 0)
        else:
            raise ValueError(f'Unknown pooling strategy: {self.strategy}')

        return pooled


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
