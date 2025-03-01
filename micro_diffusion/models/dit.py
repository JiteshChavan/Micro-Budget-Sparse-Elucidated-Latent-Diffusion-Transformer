from typing import List
from .modules import CaptionProjection, CrossAttention, MLP, MLPConfig, SelfAttention, T2IFinalLayer, TimeStepEmbedder
from .modules import create_norm, get_2d_sincos_pos_embed, get_mask, mask_out_token, modulate, insert_filler_masked_tokens


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from timm.models.vision_transformer import PatchEmbed

class AttentionBlockPromptEmbedding(nn.Module):
    """Attention block specifically for processing prompt embeddings.
    
    Args:
        n_embd(int) : Input and output dimension
        head_size (int): Channels per attention head
        # TODO: change name, fan_hidden multiplier
        mlp_n_hidden_mult (float): Multiplier for feed-forward network hidden dimension w.r.t input dimension
        # TODO: change name, something to do with layerwise scaling of n_embd
        fan_hidden_base_mult(int): Round feed-forward network fan_hidden up to nearest multiple of this value
        norm_eps (float): Epsilon value for layer normalization (avoid division by 0)
        use_bias (bool): whether to use bias in linear layers
    """

    def __init__ (self, n_embd, head_size, mlp_n_hidden_mult, fan_hidden_base_mult, norm_eps, use_bias):
        super().__init__()
        assert n_embd % head_size == 0, f"n_embd:{n_embd} should be divisble by head_size:{head_size}"

        self.n_embd = n_embd
        self.n_head = n_embd // head_size

        self.ln1 = create_norm ("layernorm", n_embd, eps=norm_eps)
        # Not specifying n_hidden implies qkv will project n_embd into 3xn_embd
        self.attn = SelfAttention (n_embd, self.n_head, qkv_bias=use_bias, norm_eps=norm_eps)

        self.ln2 = create_norm ("layernorm", n_embd, eps=norm_eps)
        self.ffn = FeedForwardNetwork (n_embd, n_hidden=int(n_embd*mlp_n_hidden_mult), fan_hidden_base_mult=fan_hidden_base_mult, use_bias=use_bias)
    
    def forward (self, x, **kwargs):
        x = x + self.attn (self.ln1(x))
        x = x + self.ffn (self.ln2(x))
        return x

    

class FeedForwardNetwork(nn.Module):
    """Feed-forward block with SiLU activation
        n_embd: input/output channel dimensionality
        n_hidden : hidden layer dimensionality
        fan_hidden_base_mult : make n_hidden nearest next multiple of this base_mult
        use_bias: Use bias in linear layers or not
    """
    def __init__(self, n_embd, n_hidden, fan_hidden_base_mult, use_bias):
        super().__init__()
        
        self.n_hidden = int (2*n_hidden/3)
        self.n_hidden = fan_hidden_base_mult * ((n_hidden + fan_hidden_base_mult - 1) // fan_hidden_base_mult)
        self.fc1 = nn.Linear (n_embd, self.n_hidden, bias=use_bias)
        self.fc2 = nn.Linear (n_embd, self.n_hidden, bias=use_bias)
        self.fc3 = nn.Linear(self.n_hidden, n_embd, bias=use_bias)

    def forward (self, x):
        return self.fc3 (F.silu(self.fc1(x)) * self.fc2(x))
