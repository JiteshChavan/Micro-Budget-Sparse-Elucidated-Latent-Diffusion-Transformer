from functools import partial
from typing import List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from composer.models import ComposerModel
from diffusers import AutoencoderKL
from easydict import EasyDict

# TODO: change later
from .utils import DATA_TYPES, DistLoss, UniversalTextEncoder, UniversalTokenizer, get_text_encoder_embedding_format
from . import dit as model_zoo

class LatentDiffusion (ComposerModel):
    """Latent diffusion model that generates images from text prompts using classifier free guidance
    
    This model combines a DiT (Diffusion Transformer) model for denoising image latents,
    a VAE for encoding/decoding images to/from the latent space, and a text encoder for converting text prompts
    into embeddings. It implements the EDM (Elucidated Diffusion Model) Sampling process.

    # TODO: introspect later
    Args:
        dit (nn.Module): Diffusion Transformer model
        vae (AutoencoderKL): VAE model from diffusers for encoding/decoding images.
        text_encoder (UniversalTextEncoder): Text encoder for converting text prompts into embeddings
        tokenizer (UniversalTokenizer): Tokenizer for tokenizing text prompts
        image_key (str, optional): Key for image data in batch dict. Defaults to 'image'
        text_key (str, optional): Key for text data in batch dict. Defaults to 'captions'
        image_latents_key (str, optional): key for precomputed image latents in batch dict. Defaults to 'image_latents'
        text_latents_key (str, optional): Key for precomputed text latents in batch dict. Defaults to 'caption_latents'
        latents_precomputed (bool, optional): Whether to use precomputed latents (must be in the batch). Defaults to True.
        dtype (str, optional): Data type for model ops. Defaults to 'bfloat16'.
        latent_res (int, optional): Resolution of latent space assuming 8x downsampling by VAE. Defaults to 32.
        p_mean (float, optional): EDM log-normal noise mean. Defaults to -0.6.
        p_std (float, optional): EDM log-normal noise standard-deviation. Defaults to 1.2.
        train_mask_ratio (float, optional): Ratio of patches to mask during training. Defaults to 0.
    """

    def __init__(
        self,
        dit: nn.Module,
        vae: AutoencoderKL,
        text_encoder: UniversalTextEncoder,
        tokenizer: UniversalTokenizer,
        image_key: str= 'image',
        text_key: str= 'captions',
        image_latents_key: str = "image_latents",
        text_latents_key: str = "caption_latents",
        latents_precomputed: bool = True,
        dtype: str = 'bfloat16',
        latent_res: int = 32,
        p_mean: float = -0.6, #t_mean for EDM time schedule
        p_std: float = 1.2,
        train_mask_ratio: float = 0 
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.image_key = image_key
        self.text_key = text_key
        self.image_latents_key = image_latents_key
        self.text_latents_key = text_latents_key
        self.latents_precomputed = latents_precomputed
        self.dtype = dtype
        self.latent_res = latent_res
        
        # config for edm formulatio of diffusion 
        # TODO: Introspect later
        self.edm_config = EasyDict({
            'sigma_min' : 0.002,
            'sigma_max' : 80,
            'p_mean' : p_mean,
            'p_std' : p_std,
            'sigma_data' : 0.9,
            'num_steps' : 18,
            'rho' : 7,
            'S_churn' : 0,
            'S_min' : 0,
            'S_max' : float('inf'),
            'S_noise' : 1
        })

        self.train_mask_ratio = train_mask_ratio
        self.eval_mask_ratio = 0 # no masking during inference/sampling/evaluation
        assert self.train_mask_ratio >= 0, f"Masking ratio has to be non negative"

        # TODO:remove later
        self.randn_like = torch.randn_like
        self.latent_scale = self.vae.config.scaling_factor

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        # Freeze vae and text_encoder
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_ (False)

        # dont FSDP wrap frozen models
        self.text_encoder._fsdp_wrap = False
        self.vae._fsdp_wrap = False
        self.dit._fsdp_wrap = True

    
    def forward (self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get image latents
        if self.latents_precomputed and self.image_latents_key in batch:
            # Assuming that latents have already been scaled, i.e., multiplied with the scaling factor
            latents = batch[self.image_latents_key]
        else:
            with torch.no_grad():
                images = batch[self.image_key]
                latents = self.vae.encode(
                    images.to(DATA_TYPES[self.dtype])
                )['latent_dist'].sample().data
                latents *= self.latent_scale
        
        # get text latent embeddings
        if self.latents_precomputed and self.text_latents_key in batch:
            conditioning = batch[self.text_latents_key]
        else:
            captions = batch[self.text_key]
            # captions are stacked as (B, 1, 77)
            captions = captions.view(-1, captions.shape[-1]) #(B, 77) as input to encoder
            # for t5
            if 'attention_mask' in batch:
                conditioning = self.text_encoder(
                    captions, attention_mask=batch['attention_mask'].view(-1, captions.shape[-1])
                )[0] # get conditioning from the tuple
            else:
                conditioning = self.text_encoder.encode(captions)[0] # extract from tuple
        
        # classifier free guidance
        # Zero out dropped captions.
        # TODO: introspect where pcond event is rolled
        if 'drop_caption_mask' in batch.keys():
            # suppose batch size 4, drop_caption_mask will look like [0, 1, 1, 0]
            # these vectors have to be propagated along each dimension of latent representation
            # latent representation would look like [B, 1, 77, 1024] for openclip/hf-14
            # propagated mask looks like (B, 1, 1, 1)
            propagated_drop_caption_vectors =  batch['drop_caption_mask'].view ([-1] + [1] *(len(conditioning.shape) - 1))

            conditioning *= propagated_drop_caption_vectors

            loss = self.edm_loss (
                latents.float(),
                conditioning.float(),
                mask_ratio = self.train_mask_ratio if self.training else self.eval_mask_ratio
            )

            # TODO: why return latents
            return (loss, latents, conditioning)
        
        # TODO: introspect and change variables later
        def model_forward_wrapper(
            self,
            x: torch.Tensor,
            sigma: torch.Tensor, # sigma_t
            y: torch.Tensor,
            model_forward_fxn: callable,
            mask_ratio: float,
            **kwargs
        ) -> dict:
            """Wrapper for the model call in EDM (https://github.com/NVlabs/edm/blob/main/training/networks.py#L632)"""
            sigma = sigma.to(x.dtype).reshape (-1, 1, 1, 1)
            c_skip = (
                self.edm_config.sigma_data ** 2 /
                (sigma ** 2 + self.edm_config.sigma_data ** 2)
            )

            c_out = sigma * self.edm_config.sigma_data / (sigma ** 2 + self.edm_config.sigma_data ** 2).sqrt()
            c_in = 1 / (self.edm_config.sigma_data ** 2 + sigma ** 2).sqrt()

            c_noise = sigma.log() / 4


            out = model_forward_fxn (
                (c_in * x).to(x.dtype),
                c_noise.flatten(),
                y,
                mask_ratio=mask_ratio,
                **kwargs
            )

            F_x = out ['sample']
            c_skip = c_skip.to(F_x.device)
            x = x.to(F_x.device)
            c_out = c_out.to(F_x.device)
            D_x = c_skip * x + c_out * F_x
            out ['sample'] = D_x
            return out