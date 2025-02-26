from typing import Any, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class MLPConfig:
    fan_in : int
    fan_h : int
    fan_out : int
    # non linearity is not optional, if not specified it will be GELU
    non_linearity : Any = nn.GELU(approximate='tanh')
    # optional
    norm_layer : Optional[Any] = None
    bias : bool = True

class SelfAttention (nn.Module):
    def __init__(self, n_embd, n_head, qkv_bias:bool=True, n_hidden=None, norm_eps=1e-6):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd

        if n_hidden is None:
            n_hidden = n_embd
        assert n_hidden % n_head == 0, f"Latent n_embd for attention layers should be divisble by n_head"
        self.n_hidden = n_hidden
        
        self.attn = nn.Linear (self.n_embd, 3*self.n_hidden, bias=qkv_bias)
        self.proj = nn.Linear (self.n_hidden, self.n_embd, bias=qkv_bias)
        self.proj.NANO_GPT_SCALE_INIT = 1 # courtesy of Andrej Karpathy; control growing variance from residual connections

        self.ln_q = create_norm ("layernorm", dim=n_hidden, eps=norm_eps)
        self.ln_k = create_norm ("layernorm", dim=n_hidden, eps=norm_eps)

    # self attention on (B, T, C)
    def forward (self, x):
        # extract shapes
        B, T, C = x.shape
        
        # alternative way
        #qkv = self.attn(x).view (B, T, 3, self.n_head, C//self.n_head)
        #q,k,v = qkv.unbind(2)
        qkv = self.attn (x) # (B, T, 3C')
        q, k, v = qkv.split (self.n_hidden, dim=2) # 3x(B, T, C')

        q = self.ln_q(q) # (B, T, C')
        k = self.ln_k(k) # (B, T, C')

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view (B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention (q, k, v, is_causal=False) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_hidden) # (B, T, C')

        y = self.proj (y) # (B, T, C)
        return y
    




class MLP (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear (config.fan_in, config.fan_h, bias=config.bias)
        self.activation = config.non_linearity
        self.mlp_norm = config.norm_layer if config.norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear (config.fan_h, config.fan_out, bias=config.bias)

    def forward (self, x: torch.Tensor)-> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.mlp_norm(x)
        x = self.fc2(x)
        return x

# type hints are mere formality, python doesnt enforce types
def create_norm (norm_type: str, dim: int, eps: float=1e-6)->nn.Module:
    """Creates a normalization layer based on specified type"""
    if norm_type == "layernorm":
        # elementwise_affine = False gives RMSNorm (NoParam Norm: no one cares about that)
        return nn.LayerNorm (dim, eps=eps, bias=False)
    else:
        raise ValueError('Norm Type Not supported!')


def modulate (x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale and Shift the input tensor"""
    # TODO: fix later
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)