import math
from collections.abc import Iterable
from itertools import repeat
from typing import Optional, Tuple, Dict, Union, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchmetrics import Metric
import open_clip
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    # remove later if redundant
    T5EncoderModel,
    T5Tokenizer
)

    
DATA_TYPES = {
    'float16' : torch.float16,
    'bfloat16' : torch.bfloat16,
    'float32' : torch.float32
}

def modulate (x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale and Shift the input tensor"""
    # TODO: fix later
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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



def get_text_encoder_embedding_format(tokenizer_name: str) -> Tuple[int, int]:
    """Returns sequence length and token embedding dimension for text encoder."""
    if tokenizer_name in [
        'stabilityai/stable-diffusion-2-base',
        'runwayml/stable-diffusion-v1-5',
        'CompVis/stable-diffusion-v1-4'
    ]:
        return 77, 1024
    if tokenizer_name in ['openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378']:
        return 77, 1024
    if tokenizer_name in ["DeepFloyd/t5-v1_1-xxl"]:
        return 120, 4096
    raise ValueError(f'encoder : {tokenizer_name} not supported')
    
class simple_2_hf_tokenizer_wrapper:
    """Simple wrapper to make OpenCLIP tokenizer match HuggingFace interface.
    
    Args:
        tokenizer (Any): OpenCLIP tokenizer instance
    """
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.model_max_length = tokenizer.context_length
        
    def __call__(self, caption: str, padding: str = 'max_length', max_length: Optional[int] = None, truncation: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
        tokenized_caption = {'caption_idx' : self.tokenizer(caption, context_length=max_length)}
        return tokenized_caption

class UniversalTokenizer:
    """Universal tokenizer supporting multiple model types.
        Args:
        name (str): Name/path of the tokenizer to load
    """

    def __init__(self, name: str):
        self.name = name
        seq_length, n_embd = get_text_encoder_embedding_format (name)
        if self.name.startswith("openclip:"):
            self.tokenizer = simple_2_hf_tokenizer_wrapper(open_clip.get_tokenizer (name.lstrip('openclip:')))

            assert seq_length == self.tokenizer.model_max_length, "sequence length doesnt match, to specs that we wrote in custom function"
        elif self.name == "DeepFloyd/t5-v1_1-xxl":
            # for T5 we use context smaller than max_seq_length (from the get_text_encoder_embedding_fromat) hence no assertion
            self.tokenizer = T5Tokenizer.from_pretrained(name)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(name, subfolder='tokenizer')
            assert seq_length == self.tokenizer.model_max_length, "sequence length doesnt match, to specs that we wrote in custom function"
        # in case of T5 override model_max_length with what we are using
        self.model_max_length = seq_length
    

    def tokenize(self, captions: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Takes input string or list of strings, tokenizes the caption by padding upto max length, truncating sequences
        if length exceeds max length, returns tokenized format as pt (pytorch tensors), returns attention mask as well
        which marks padded tokens with 0s"""
        if self.name == "DeepFloyd/t5-v1_1-xxl":
            text_tokens_and_mask = self.tokenizer(captions, padding='max_length', max_length=self.model_max_length, truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
            mask_and_tokens = {'input_idx': text_tokens_and_mask['input_ids'], 'attention_mask': text_tokens_and_mask['attention_mask']}
            return mask_and_tokens
        
        else:
            # Avoid attention mask for CLIP tokenizers as they are not used
            # the calls usually return a dictionary
            tokenized_caption = self.tokenizer( captions, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            return {'input_idx': tokenized_caption['input_ids']}
        
class UniversalTextEncoder(nn.Module):
    """Universal text encoder supporting multiple model types.
    
    Args:
        name (str): Name/path of the model to load
        dtype (str): Data type of model weights
        pretrained (bool, True): wheter to load pretrained weights
    """
    def __init__(self, name, weights_dtype, pretrained=True):
        super().__init__()
        self.name = name
        if self.name.startswith("openclip:"):
            assert pretrained, f"load default pretrained model from openclip"
            self.encoder = openclip_text_encoder(
                open_clip.create_model_and_transforms(name.lstrip('openclip:'))[0],
                torch_dtype= DATA_TYPES[weights_dtype]
            )
        elif self.name == "DeepFloyd/t5-v1_1-xxl":
            self.encoder = T5EncoderModel.from_pretrained (
                name,
                torch_dtype=DATA_TYPES[weights_dtype],
                pretrained = pretrained
            )
        else:
            self.encoder = CLIPTextModel.from_pretrained (
                name,
                subfolder='text_encoder',
                torch_dtype=DATA_TYPES[weights_dtype],
                pretrained=pretrained
            )
    
    def encode (self, tokenized_caption: torch.Tensor, attention_mask=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.name == "DeepFloyd/t5-v1_1-xxl":
            out = self.encoder(tokenized_caption, attention_mask=attention_mask)['last_hidden_state']
            out = out.unsqueeze(dim=1)
            return out, None
        else:
            return self.encoder(tokenized_caption)


class openclip_text_encoder (nn.Module):
    """OpenCLIP text encoder abstraction
    
    Args:
        clip_model (Any): OpenCLIP model instance
        weights_dtype (torch.dtype, torch.float32): Data type for model weights
    """

    def __init__ (self, clip_model, weights_dtype=torch.float32, **kwargs):
        super().__init__()
        self.clip_model = clip_model
        
        # TODO: inspect later
        self.device = None
        self.weights_dtype = weights_dtype

    def forward_fn (self, text: torch.Tensor)-> Tuple[torch.Tensor, None]:
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding (text).to(cast_dtype) # (B, T, C)
        # context length is constant as seq_length as seen in one of the functions above (77)
        # dont need to torch.arange (0, len(x)).to(dtype).to(device)
        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        # No other option but to follow HF convention
        # Sucks!
        x = x.permute(1, 0, 2) # B, T, C -> T, B, C
        x = self.clip_model.transformer (x, attn_mask=self.clip_model.attn_mask)
        x = x.permute (1, 0, 2) # T, B, C -> B, T, C
        x = self.clip_model.ln_final(x) # (B, T, C)
        x = x.unsqueeze (dim=1) # (B, 1, T, C) expected for text_emb
        return x, None # HF encoders expect to return multiple values with first being text_emb
    
    def forward (self, text, **kwargs)-> Tuple[torch.Tensor, None]:
        with torch.autocast(device_type='cuda', dtype=self.weights_dtype):
            return self.forward_fn(text)

