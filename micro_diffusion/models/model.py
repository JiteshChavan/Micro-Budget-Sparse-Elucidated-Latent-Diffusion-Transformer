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

            # implicit broadcasting:
            # (512, 1, 77, 1024) * (512, 1, 1, 1)
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
        sigma_t: torch.Tensor, # sigma_t comes from logvar
        y: torch.Tensor,
        model_forward_fxn: callable,
        mask_ratio: float,
        **kwargs
    ) -> dict:
        """Wrapper for the model call in EDM (https://github.com/NVlabs/edm/blob/main/training/networks.py#L632)"""
        # shape of signal x0 = (B, 4, 32, 32)
        # can be done like this sigma_t = sigma_t.unsqueeze(1,2,3)
        sigma_t = sigma_t.to(x.dtype).reshape (-1, 1, 1, 1)
        c_skip = (
            self.edm_config.sigma_data ** 2 /
            (sigma_t ** 2 + self.edm_config.sigma_data ** 2)
        )

        c_out = sigma_t * self.edm_config.sigma_data / (sigma_t ** 2 + self.edm_config.sigma_data ** 2).sqrt()
        c_in = 1 / (self.edm_config.sigma_data ** 2 + sigma_t ** 2).sqrt()

        # noise scaling
        # sigma_t shape (B, 1, 1, 1)
        c_noise = sigma_t.log() / 4 # (B, 1, 1, 1)

        out = model_forward_fxn (
            (c_in * x).to(x.dtype),
            c_noise.flatten(), # TODO: introspect later (B)
            y, # text caption?
            mask_ratio=mask_ratio, 
            **kwargs
        )

        # TODO: change sample keys
        F_x = out ['sample']
        c_skip = c_skip.to(F_x.device)
        x = x.to(F_x.device)
        c_out = c_out.to(F_x.device)
        D_x = c_skip * x + c_out * F_x
        out ['sample'] = D_x
        return out
    
    def edm_loss (self, x: torch.Tensor, y: torch.Tensor, mask_ratio: float = 0, **kwargs) -> torch.Tensor:
        # sample B (512) eps from N(0, I) reshape to be of shape (B, 1, 1, 1)
        eps = torch.randn([x.shape[0], 1, 1, 1], device=x.device) # eps from N(0, I) (B, 1, 1, 1)
        
        # sample sigma(t) from lognormal (P_mean, P_std)
        sigma_t = (self.edm_config.P_mean + self.edm_config.P_std*eps).exp()
        
        # loss weighting 
        loss_weight = (sigma_t ** 2 + self.edm_config.sigma_data ** 2) / (sigma_t * self.edm_config.sigma_data) ** 2

        # sample (B, 4, 32, 32) random noise from N(0, I) to add unto x
        noise_eps = torch.randn_like (x) * sigma_t # (B, 4, 32, 32) * (B, 1, 1, 1)

        # x_t (corrupted signal x0 + sigma_t eps from (0,I))
        # note that we dont scale down signal with root 1-beta_t as c_in scaling makes
        # inputs to neural network unit variance, no need to complicate the SDE formulation
        corrupted_signal = x + noise_eps

        model_out = self.model_forward_wrapper (corrupted_signal, sigma_t, y, self.dit, mask_ratio=mask_ratio, **kwargs)
        # change key later, bullshit
        D_xn = model_out['sample'] # (B, C, H, W)
        loss = loss_weight * ((D_xn - x)**2) # (B, C, H, W)
        
        # TODO: introspect: I think we finetune on 0 mask ratio after pretraining
        if mask_ratio > 0:
            # if image_latents were masked before feeding to DiT to reduce input seq length to
            # transformer, the 8x8 has to be converted to 32x32 then fed into VAE decoder

            assert (self.dit.training and 'mask' in model_out), f"Masking is only done during training"
            loss = loss.mean(dim=1) # mean along channels shape will be (B, 32, 32) = (B, H, W)
            # convert to DiT semantic space, in terms of patches
            loss = F.avg_pool2d (loss, self.dit.patch_size) # (B, 16, 16) loss for all patches processed by DiT
            # unroll the patches and then remove impact of masked patches, because the masked patches do not
            # contribute to DiT loss, because input to DiT is only unmasked patches and it only learns to
            # denoise unmasked patches
            # unrolling makes it easier because mask is of shape (B, transformer_sequence_length) where 
            # transformer_sequence_length = 32/dit.patch_size * 32/dit.patch_size; 32 comes from latent res for vae

            loss = loss.flatten (1) # (B, 256); preserve B and 256 comes from patch_size = 2 and vae latent res = 32

            unmask = 1 - model_out['mask'] # (B, 256) 1 where unmasked, 0 where patches were masked
            # for average loss coressponding to all patches, we only need divide by number of unmasked patches
            normalization_factor = unmask.sum(dim = 1) # (B, 1)
            loss = (loss * unmask).sum(dim = 1) # (B, 1)
            loss /= normalization_factor # (B, 1)
        return loss.mean()

    # Composer specific formatting of model loss and eval functions.
    def loss (self, outputs: tuple, batch: dict)-> torch.Tensor:
        # forward pass already computed the loss function
        return outputs[0]
    
    # Composer specific validation format
    def eval_forward (self, batch:dict, outputs:Optional[tuple] = None)-> tuple:
        # Skip if output is already calculated (e.g, during training forward pass)
        if outputs is not None:
            return outputs
        else:
            loss, _, _ = self.forward (batch)
            return loss, None, None
    
    # TODO: introspect later, get reduced dist loss for validation maybe?
    def get_metrics (self, is_train:bool = False) -> dict:
        # get_metrics is expected to return a dict is composer
        return {'loss' : DistLoss()}
    
    def update_metric (self, batch: dict, outputs: tuple, metric: DistLoss):
        metric.update (outputs[0])
    
    @torch.no_grad()
    def edm_sampler_loop(self, x:torch.Tensor, y:torch.Tensor, steps: Optional[int] = None, cfg: float = 1.0, **kwargs)->torch.Tensor:
        # no masking during inference
        mask_ratio = 0
        # we have to forward model twice for CFG, once with conditioning, once without
        model_forward_fxn = (partial(self.dit.forward, cfg=cfg) if cfg > 1.0 else self.dit.forward)

        # timestep discretization
        num_steps = self.edm_config.num_steps if steps is None else steps
        step_indices = torch.arange (num_steps, dtype=torch.float16, device=x.device)

        t_steps = (
            self.edm_config.sigma_max ** (1 / self.edm_config.rho)
        )