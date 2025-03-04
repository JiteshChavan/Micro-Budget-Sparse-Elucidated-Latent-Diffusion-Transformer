from typing import Any, Optional, Union, Tuple
from collections.abc import Iterable
from itertools import repeat
import numpy as np

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
    
    def custom_init(self, init_std:float):
        nn.init.trunc_normal_(self.attn.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)

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
    
    def custom_init(self, init_std: float)-> None:
        for linear in (self.cx_attn_q, self.cx_attn_kv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)


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
    def __init__(self, n_embd, sigma_t_embd, patch_size, fan_out, activation, norm_final):
        super().__init__()
        self.n_embd = n_embd
        self.sigma_t_embd = sigma_t_embd
        self.linear = nn.Linear (n_embd, patch_size*patch_size*fan_out)
        self.adaLN_modulation = nn.Sequential (
            activation,
            nn.Linear (sigma_t_embd, 2 * n_embd)
        )
        self.norm_final = norm_final
    """
        x is image (B, T, C)
        time_embd (B, C) -> time embedding doesnt have tokens or sequence length
        just one embedding for one timestep for one image in one batch
    """
    def forward (self, x, time_embd):
        # get scale and shift from adaLN depending on current timestep
        shift, scale = self.adaLN_modulation (time_embd).split(self.n_embd, dim=-1)
        # normalize then modulate x with respect to current time step
        x = modulate (self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x

class TimeStepEmbedder (nn.Module):
    """
        Embeds scalar noise level sigma_t into freq_embd dimensional vector using sinusoidal embeddings
        then refines this freq_embd dimensional vectors using MLP to prouce n_embd dimensional representation
        that is compatible with DiT
    """

    def __init__ (self, n_hidden, activation, freq_embd:int = 512):
        super().__init__()
        self.freq_embd = freq_embd
        self.n_hidden = n_hidden
        self.activation = activation
        self.mlp = nn.Sequential (
            nn.Linear (freq_embd, n_hidden, bias=True),
            self.activation,
            nn.Linear (n_hidden, n_hidden, bias=True)
        )
    
    @staticmethod
    def embed_timestep (sigma_t, freq_embd, max_period = 10000):
        # half the time embedding dimensions will be cosine rest of half will be sine
        # if freq_embd is odd, then we append 0 frequency component to maintain consistency
        half = freq_embd // 2

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

        sigma_t = sigma_t.unsqueeze(-1).float() #(1, 1)
        args = sigma_t * freqs # (1,1) * (half)
        # args is now (1, half)
        # embedding vector thats freq_embd long if freq_embd is even
        # otherwise its freq_embd - 1
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)

        if freq_embd % 2 != 0:
            # rectify the dimensionality of the embedding if freq_embd was odd
            # freq_embd - 1 -> freq_embd by appending 0 frequency component
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, 0])), dim=-1)
        return embedding #(1, freq_embd)

    # return type of first parameter in this class
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    def forward (self, sigma_t):
        t_embedding = self.embed_timestep (sigma_t, self.freq_embd).to(self.dtype)
        return self.mlp (t_embedding)


def ntuple(n_dim :int, x):
    """ Converts input into n_dim-tuple. For handling resolutions"""
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    else:
        return tuple(repeat(x, n_dim))

def get_2d_sincos_pos_embed (
        n_embd, 
        grid_size:Union[int, Tuple[int, int]], 
        base_size:int=16, 
        cls_token :bool = False, extra_tokens:int=0,
        pos_interp_scale = 1.0
    ):
        """ One spot / pixel in grid is represented by n_embd channels, n_embd/2 come from scaled x
            coordinate, n_embd/2 come from scaled y co-ordinate
        """
        if isinstance(grid_size, Iterable) and not isinstance(grid_size, str):
            grid_size = ntuple (2, grid_size)
        # interpolate position embeddings to adapt model to different resolutions.
        # makes it so that specific spatial positions have similar embeddings
        
        # height is 0, width is 1
        # division by grid_size[0] makes pos embeddings for different resolutions the same at specific spots
        # further division by (base_size / pos_interp_scale) mutates the range (say 0 to 16 instead of 0 to 1)
        # so that model can distinguish better
        grid_h = np.arange (grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / pos_interp_scale
        grid_w = np.arange (grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / pos_interp_scale

        # width, height
        grid = np.meshgrid (grid_w, grid_h)

        # stack along axis 0 to get two matrices, first for width co-ordinates second for height co-ordinates

        grid = np.stack (grid, axis=0) # (2, grid_size[1], grid_size[0]) (2, W, H)
        # make (2,1,W,H)
        grid = grid.reshape (2, 1, grid_size[1], grid_size[0]) # add spurious dimension to be processed by get_embedding function

        pos_embedding = get_2d_sinusoidal_embedding_from_grid (n_embd, grid)
        if cls_token and extra_tokens > 0:
            pos_embedding = np.concatenate ([np.zeros([extra_tokens, n_embd]), pos_embedding], axis=0)
        
        return pos_embedding # (HW, n_embd)

def get_2d_sinusoidal_embedding_from_grid (n_embd, grid):
    "Takes in a grid (2, 1, W, H), generates 2D embedding for the said grid"

    assert n_embd % 2 == 0
    half_embd = n_embd // 2

    # send x co-ordinates (x) (1, W, H)
    embd_h = get_1d_sinusoidal_embedding(half_embd, grid[0]) # (HW, half_embd)
    # send y co-ordinates (y) (1, W, H) 
    embd_w = get_1d_sinusoidal_embedding(half_embd, grid[1]) # (HW, half_embd)
    positional_embedding = np.concatenate([embd_h, embd_w], axis = 1) # (HW, n_embd)
    return positional_embedding 

def get_1d_sinusoidal_embedding (n_embd, pos):
    """1D sinusoidal embeddings from grid"""
    assert n_embd % 2 == 0

    omega = np.arange(n_embd//2, dtype=np.float64) # (D/2)
    omega = omega / (n_embd//2)

    # omega is linearly spaced, to generate exponentially decaying freqs
    # exponentiate omge to a number thats smaller than 1
    freqs = (1/10000) ** omega # (D/2)
    pos = pos.reshape(-1) # (M)
    # now inject position into these frequencies
    # first way: if they are tensors
    # pos.unsqueeze(1) * freqs
    # second way
    mutated_freqs = np.einsum('m,d->md', pos, freqs) # (HW=M, D/2)

    sin_embd = np.sin(mutated_freqs)
    cos_embd = np.cos(mutated_freqs)
    return np.concatenate ([sin_embd, cos_embd], axis=1)

def get_mask (batch, length, mask_ratio, device):
    # calculate fraction thats not meant to be masked
    assert (mask_ratio <= 1.0 and mask_ratio >= 0.0), f"invalid mask ratio"

    keep_ratio = 1 - mask_ratio
    length_keep = int (keep_ratio * length)

    # sample noise for randomized masking
    noise = torch.rand (batch, length, device=device) # (B, T)
    # Find indices that correspond to smaller noise along seq length
    idx_ascending = torch.argsort (noise, dim = 1) # keep small, remove high

    # Find indices into idx_asc that yield original noise when indexed into noise
    # basically noise[idx_asc[idx_restore[0]]] gives the original value at 0 for the sampled noise
    idx_restore = torch.argsort (idx_ascending, dim = 1)
    # keep only unmasked indices
    idx_keep = idx_ascending[:, :length_keep] # (B, input_to_DIT)

    mask = torch.ones (batch, length, device=device)
    mask[:, :length_keep] = 0
    # gather mask with respect to the originally smapled random noise to make it random masking
    mask = torch.gather (mask, dim=1, index=idx_restore)

    return {
        'mask' : mask,         # (B, T)
        'idx_keep' : idx_keep, # (B, input_to_DIT=length_keep)
        'idx_restore' : idx_restore # (B, T)
    }

def mask_out_token (x, idx_keep):
    """Mask out tokens specified by idx keep
        idx_keep (B, length_keep)
        x (B, T, C)
    """
    B, T, C = x.shape
    # mutate idx_keep to preserve B, T and propagate it along all the n_embd dimensions
    # idx_keep shape is [n] (n = DIT_in_length = 1- mask_ratio% of input tokens) 
    gather_index = idx_keep.unsqueeze(-1).repeat(1, 1, C) # repeat (B, length_keep, 1) once along batch, once along T, n_embed times along C
    x_masked = torch.gather (x, dim=1, index=gather_index) # gather_index (B, length_keep, C)
    # x_masked will be B, DIT_in_length, C, which will be input to our DIT

    return x_masked

def fill_out_masked_tokens(x:torch.Tensor, stub_token:torch.Tensor, idx_restore:torch.Tensor):
    """
        x -> B, keep_length, C
        filler_mask_token -> 1, 1, C just a stub
        idx_restore -> B, T (which indices to index into to get a sequence that respects original noise
        which we sorted and selected first keep_length indices)
    """
    masked_out_length = idx_restore.shape[1]-x.shape[1] # the tokens that DIT doesnt even see, 75% of og tokens
    filler_mask_tokens = stub_token.repeat (x.shape[0], masked_out_length, 1)
    
    x_og_kappa = torch.cat ((x, filler_mask_tokens), dim=1)
    restore_gather_index = idx_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    x_og_kappa = torch.gather (x_og_kappa, dim=1, index=restore_gather_index)
    return x_og_kappa


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
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.02)
        
        if hasattr(self.mlp_norm, "reset_parameters"):
            self.mlp_norm.reset_parameters()

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
