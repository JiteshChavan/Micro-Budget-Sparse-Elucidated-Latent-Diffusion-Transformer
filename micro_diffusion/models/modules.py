from typing import Any, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass

# Remember to write _init_weights (kaiming init for all layers in DIT), Latent Diffusion model script has no
# weights to be initialized

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

        q = q.view(B, T, self.n_head, self.n_hidden // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.n_hidden // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view (B, T, self.n_head, self.n_hidden // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention (q, k, v, is_causal=False) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_hidden) # (B, T, C')
        
        # project back to n_embd from n_hidden
        y = self.proj (y) # (B, T, C)
        return y
    

class CrossAttention (nn.Module):
    # B T C text k v
    # B T X image q
    # channels have to be same for q @ k
    def __init__(self, n_embd, n_head, n_hidden=None, norm_eps=1e-6, qkv_bias=True):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head

        if n_hidden is None:
            n_hidden = n_embd
        self.n_hidden = n_hidden

        assert n_hidden % n_head == 0, f"n_hidden:{n_hidden} should be divisible among heads:{n_head}"

        self.ln_q = create_norm ("layernorm", dim=n_hidden, eps=norm_eps)
        self.ln_k = create_norm ("layernorm", dim=n_hidden, eps=norm_eps)
        
        # query from image
        self.cx_attn_q = nn.Linear (n_embd, n_hidden, bias=qkv_bias)
        self.cx_attn_kv = nn.Linear (n_embd, 2*n_hidden, bias=qkv_bias)

        self.proj = nn.Linear (n_hidden, n_embd, bias=qkv_bias)
        self.proj.NANO_GPT_SCALE_INIT = 1
    
    def forward (self, x, condition):
        # shapes from image
        B, T, C = x.shape
        # shapes from text
        T_B, T_T, T_C = condition.shape

        assert C == T_C, f"channels mismatch in cross attention"

        # kv from text
        kv = self.cx_attn_kv (condition) # (B, T, 2C') C' = n_hidden
        # q from image x
        q = self.cx_attn_q (x) # (B, T, C')

        k, v = kv.split (self.n_hidden, dim = 2) # 2x (B, T, C')
        # normalize key and query
        q = self.ln_q (q) # (B, T, C')
        k = self.ln_k (k) # (B, T, C')

        q = q.view (B, T, self.n_head, self.n_hidden // self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view (T_B, T_T, self.n_head, self.n_hidden // self.n_head).transpose(1,2) # (T_B, nh, T_T, hs)
        v = v.view (T_B, T_T, self.n_head, self.n_hidden // self.n_head).transpose(1,2) # (T_B, nh, T_T, hs)

        # Q @ K 
        # B nh T hs @ T_B, nh, T_T, hs
        # B nh T T_T this @ value (T_B, nh, T_T, hs)
        # B nH T hs

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # (B, nh, T, hs)
        y = y.transpose (1,2).contiguous().view(B, T, self.n_hidden) # (B, T, C')
        y = self.proj(y) # (B, T, C')-> (B, T, C)
        return y


class CaptionProjection (nn.Module):
    """ Projects caption embeddings to model n_embd
        specify proper MLP config at init
    """
    # NANO_GPT_SCALE_INIT not required no residual pathway
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = MLP (config)
        # TODO: introspect for residual connection later
        #self.proj.fc2.NANO_GPT_SCALE_INIT = 0
    
    def forward (self, x):
        x = self.proj(x)
        return x

class T2IFinalLayer (nn.Module):
    """
        # Step 1: scale and shift (Modulate) image signal x adaptively based on time_embedding
        # step 2: expand one block (1, C) into (1, patchsize * patchsize * C)
        # implicitly expanding one block into patchsize * patchsize blocks which can be reshaped/arranged
        # to processed by VAE decoder

        adaLN stands for adaptive layernorm, it devises gamma/beta as a linear function of input
        so every input has its own gamma/beta
        which is then used to normalize the inputs as  (gamma * input + beta)

    """
    # cant hardcode n_hidden because layerwise scaling increases dit n_embd as we progress through the network
    def __init__(self, n_hidden, patch_size, fan_out, activation, norm_final):
        super().__init__()
        self.n_hidden = n_hidden
        self.linear = nn.Linear (n_hidden, patch_size*patch_size*fan_out)
        self.adaLN_modulation = nn.Sequential (
            activation,
            nn.Linear (n_hidden, 2 * n_hidden)
        )
        self.norm_final = norm_final
    """
        x is image (B, T, C)
        time_embd (B, C) -> time embedding doesnt have tokens or sequence length
        just one embedding for one timestep for one image in one batch
    """
    def forward (self, x, time_embd):
        # get scale and shift from adaLN depending on current timestep
        shift, scale = self.adaLN_modulation (time_embd).split(self.n_hidden, dim=1)
        # normalize then modulate x with respect to current time step
        x = modulate (self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x

class TimeStepEmbedder (nn.Module):
    """
        Embeds scalar noise level sigma_t into n_t_embd dimensional vector using sinusoidal embeddings
        then refines this n_t_embd dimensional vectors using MLP to prouce n_embd dimensional representation
        that is compatible with DiT
    """

    def __init__ (self, sigma_t, n_t_embd, n_embd, activation):
        self.sigma_t = sigma_t
        self.n_t_embd = n_t_embd
        self.n_embd = n_embd
        self.activation = activation
        self.mlp = nn.Sequential (
            nn.Linear (n_t_embd, n_embd, bias=True),
            self.activation,
            nn.Linear (n_embd, n_embd, bias=True)
        )
    
    @staticmethod
    def embed_timestep (sigma_t, n_t_embd, max_period = 10000):
        # half the time embedding dimensions will be cosine rest of half will be sine
        # if n_t_embd is odd, then we append 0 frequency component to maintain consistency
        half = n_t_embd // 2

        log_freqs = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=sigma_t.device)
        # now we have log_freqs in descending order, they start from 0 and grow negative as we move right
        # later we exponentiate log_freqs to get frequencies, and exp(verylarge negative number) is very small
        # we dont want frequency components that are really close to 0 and almost indistinguishable from each other at the tail end
        # thats why we scale down these negative numbers by "half" so that we have smaller negative numbers
        log_freqs = log_freqs / half
        # exponentiate to get freqs
        freqs = torch.exp (log_freqs)
        # note that we resort to exponential scale /decay to ensure wide range of dfs
        # df would be constant in linear scale

        sigma_t = sigma_t.unsqueeze(1) #(1, 1)
        args = sigma_t * freqs
        # embedding vector thats n_t_embd long if n_t_embd is even
        # otherwise its n_t_embd - 1
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)

        if n_t_embd % 2 != 0:
            # rectify the dimensionality of the embedding if n_t_embd was odd
            # n_t_embd - 1 -> n_t_embd by appending 0 frequency component
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, 0])), dim=-1)
        return embedding

    # return type of first parameter in this class
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    def forward (self, sigma_t):
        t_embedding = self.embed_timestep (sigma_t, self.n_t_embd).to(self.dtype)
        return self.mlp (t_embedding)


class MLP (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear (config.fan_in, config.fan_h, bias=config.bias)
        self.activation = config.non_linearity
        self.mlp_norm = config.norm_layer if config.norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear (config.fan_h, config.fan_out, bias=config.bias)
        # TODO: introspect for res con later
        #self.fc2.NANO_GPT_SCALE_INIT = 1

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
    # X (B, T, C)
    # scale (B,C)
    # shift (B,C)
    # return (B,T,C) * (1 + (B, [1], C)) + (B, [1], C)

    # add 1 to scale so that during intial stages, if scale is 0 it doesnt completely eliminate x
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)