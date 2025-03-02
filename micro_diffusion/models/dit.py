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
        hidden_base_mult(int): Round feed-forward network fan_hidden up to nearest multiple of this value
        norm_eps (float): Epsilon value for layer normalization (avoid division by 0)
        use_bias (bool): whether to use bias in linear layers
    """

    def __init__ (self, n_embd, head_size, mlp_n_hidden_mult, hidden_base_mult, norm_eps, use_bias):
        super().__init__()
        assert n_embd % head_size == 0, f"n_embd:{n_embd} should be divisble by head_size:{head_size}"

        self.n_embd = n_embd
        self.n_head = n_embd // head_size

        self.ln1 = create_norm ("layernorm", n_embd, eps=norm_eps)
        # Not specifying n_hidden implies qkv will project n_embd into 3xn_embd
        self.attn = SelfAttention (n_embd, self.n_head, qkv_bias=use_bias, norm_eps=norm_eps)

        self.ln2 = create_norm ("layernorm", n_embd, eps=norm_eps)
        self.ffn = FeedForwardNetwork (n_embd, n_hidden=int(n_embd*mlp_n_hidden_mult), hidden_base_mult=hidden_base_mult, use_bias=use_bias)
    
    def forward (self, x, **kwargs):
        x = x + self.attn (self.ln1(x))
        x = x + self.ffn (self.ln2(x))
        return x
    
    def custom_init (self, init_std:float =0.02):
        self.attn.custom_init(init_std=init_std)
        self.ffn.custom_init(init_std=init_std)

class FeedForwardNetwork(nn.Module):
    """Feed-forward block with SiLU activation
        n_embd: input/output channel dimensionality
        n_hidden : hidden layer dimensionality
        hidden_base_mult : make n_hidden nearest next multiple of this hidden_base_mult
        use_bias: Use bias in linear layers or not
    """
    def __init__(self, n_embd, n_hidden, hidden_base_mult, use_bias):
        super().__init__()
        
        self.n_hidden = int (2*n_hidden/3)
        self.n_hidden = hidden_base_mult * ((n_hidden + hidden_base_mult - 1) // hidden_base_mult)
        self.fc1 = nn.Linear (n_embd, self.n_hidden, bias=use_bias)
        self.fc2 = nn.Linear (n_embd, self.n_hidden, bias=use_bias)
        self.fc3 = nn.Linear(self.n_hidden, n_embd, bias=use_bias)

    def forward (self, x):
        return self.fc3 (F.silu(self.fc1(x)) * self.fc2(x))
    
    def custom_init (self, init_std:float):
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        for linear in (self.fc2, self.fc3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
    
class FeedForwardECMoe (nn.Module):
    """Expert Choice style Mixture of Experts feed forward layer with GELU activation
    
    Args:
        num_experts (int) : number of experts in the layer
        expert_capacity (float): capacity factor determining tokens per expert
        n_embd (int): Input and output dimension
        n_hidden (int): hidden layer channels
        hidden_base_mult (int) : Round hidden dimension upto next nearest multiple of this value
    """

    def __init__(self, num_experts, expert_capacity:float, n_embd, n_hidden, hidden_base_mult):
        self.num_experts = num_experts

        # scaling that restricts or boosts capacity of an expert (eg: 0.5x or 1.5x) we default to 1.0 I think
        self.expert_capacity = expert_capacity
        
        self.n_embd = n_embd
        self.hidden_base_mult = hidden_base_mult
        self.n_hidden = hidden_base_mult * ((n_hidden + hidden_base_mult -1) // hidden_base_mult)

        # to get softmax over num_experts for T tokens
        self.gate = nn.Linear (n_embd, num_experts, bias=False) # bias false makes sense in case model wants to 1 hot on experts

        # each expert goes from n_embd to n_hidden
        self.w1 = nn.Parameter (torch.ones (num_experts, n_embd, n_hidden))
        # non linear activation
        self.gelu = nn.GELU()
        # each expert goes from n_hidden to n_embd
        self.w2 = nn.Parameter (torch.ones (num_experts, n_hidden, n_embd))
    
    def forward (self, x:torch.Tensor):
        # extract shapes
        assert x.dim() == 3
        B, T, C = x.shape

        
        tokens_per_expert = int( self.expert_capacity * T / self.num_experts)

        # get scores, softmax for each token over experts, how appealing is an expert to each of the T tokens
        scores = self.gate (x) # (B, T, E) E is number of experts
        probs = F.softmax (scores, dim=-1) # probs for T tokens across experts

        probs_expert_looking_at_all_tokens = probs.permute(0, 2, 1) # (B, E, T)

        # gather top-tokens-per-expert
        # probs, idices
        expert_specific_token_probs, expert_specific_tokens = torch.topk (probs_expert_looking_at_all_tokens, tokens_per_expert, dim=-1)
        # (B, E, l)       (B, E, l) l is tokens per expert
        # create one hot vectors of T size for the selected tokens, so that we can extract from B,T,C
        # to construct xin for moe
        extract_from_x_one_hot = F.one_hot(expert_specific_tokens, num_classes=T).float() # (B, E, l, T)

        # Goal: (B, E, l C) from x
        xin = torch.einsum ('BElT, BTC -> BElC', extract_from_x_one_hot, x)
        
        # forward
        activation = torch.einsum ('BElC, ECH -> BElH', xin, self.w1) # (B, E, l, H)
        activation = self.gelu(activation)
        activation = torch.einsum ('BElH, EHC -> BElC', activation, self.w2) # (B, E, l, C)

        # scale the activation with gating score probs, so that stronger experts have greater influence on the outputs
        activation = activation * expert_specific_token_probs.unsqueeze(dim=-1)

        # use inner product to combine results of T tokens from all the different experts
        out = torch.einsum ('BElC, BElT -> BTC', activation, extract_from_x_one_hot)
        return out
    
    def custom_init (self, init_std:float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)

class DiTBlock (nn.Module):
    """DiT transformer block comprising Attention and MLP blocks. It supports choosing between dense feed-forward
    and expert choice style Mixture of Experts feed forward blocks.

    Args:
        n_embd (int): Input and Output dimension of the block
        head_size (int) : Channels per attention head
        mlp_n_hidden_mult (float):  Multiplier for feed-forward network hidden dimension w.r.t input dimension
        qkv_ratio (float): Ratio for dimension in qkv layers in attention block
        hidden_base_mult (int): Round hidden dimension upto next nearest multiple of this value
        pooled_emb_dim (int): Dimension of pooled caption embeddings
        norm_eps (float): Epsilon for layer normalization
        depth_init (bool): Whether to initialize weights of the last layer in MLP/Attention block based on block index
        layer_id (int): Index of this block in dit model
        num_layers (int): total number of blocks in the dit model
        compress_cx_attn (bool): whether to scale cross attention qkv dimension using qkv_ratio
        use_bias (bool) : whether to use bias in linear layers
        moe_block (bool) : whether to use mixture of experts for MLP block
        num_experts (int) : Number of experts if using MoE block
        expert_capacity (float) : Capacity factor for each expert if using MoE block
    """
    
        








        