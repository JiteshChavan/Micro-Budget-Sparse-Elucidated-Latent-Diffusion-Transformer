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

        if self.vae is not None:
            self.latent_scale = self.vae.config.scaling_factor

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        # Freeze vae and text_encoder
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.vae is not None:            
            self.vae.requires_grad_ (False)

        # dont FSDP wrap frozen models
        if self.text_encoder is not None:
            self.text_encoder._fsdp_wrap = False
        if self.vae is not None:            
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

        # cfg defaults to 1.0 implying we always do forward without cfg,
        # but we drop captions in training set
        out = model_forward_fxn (
            (c_in * x).to(x.dtype),
            c_noise.flatten(), # TODO: introspect later (B)
            y, # text caption
            mask_ratio=mask_ratio, 
            **kwargs
        )

        # TODO: change sample keys
        F_x = out ['sample'] # output of DiT 
        c_skip = c_skip.to(F_x.device)
        x = x.to(F_x.device)
        c_out = c_out.to(F_x.device)
        D_x = c_skip * x + c_out * F_x # output of model (green box)
        out ['sample'] = D_x # approximation of x0
        return out
    
    def edm_loss (self, x: torch.Tensor, y: torch.Tensor, mask_ratio: float = 0, **kwargs) -> torch.Tensor:
        # sample B (512) eps from N(0, I) reshape to be of shape (B, 1, 1, 1)
        eps = torch.randn([x.shape[0], 1, 1, 1], device=x.device) # eps from N(0, I) (B, 1, 1, 1)
        
        # sample sigma(t) from lognormal (p_mean, p_std)
        sigma_t = (self.edm_config.p_mean + self.edm_config.p_std*eps).exp()
        
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
            # if image_latents were masked before feeding to DiT to reduce input seq length to DiT
            # then we have to accomodate the loss thats relevant only to active tokens (unmasked tokens)

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
        # create new callable entity.
        # nothing fancy about partial keyword

        # we use forward with cfg in sampling
        # model has learned to handle both conditional and unconditional in training so we do both
        # interpolate with the CFG formula, during sampling inference
        model_forward_fxn = (partial(self.dit.forward, cfg=cfg) if cfg > 1.0 else self.dit.forward)

        # timestep discretization
        num_steps = self.edm_config.num_steps if steps is None else steps
        step_indices = torch.arange (num_steps, dtype=torch.float16, device=x.device)

        t_steps = (
            self.edm_config.sigma_max ** (1 / self.edm_config.rho) +
            step_indices / (num_steps - 1) *
            ( (self.edm_config.sigma_min) ** (1 / self.edm_config.rho) -
            self.edm_config.sigma_max ** (1 / self.edm_config.rho)) 
        ) ** self.edm_config.rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

        # construct reverse noise schedule as interpolation between sigma_max and sigma_min
        # append a jump from t = 1 to t = 0, that samples from distribution with 0 variance
        # implicit implication that beta(t) decays, stochastic exploration dies down towards t = 0 and 
        # finally variance of weiner process becomes 0
        # Corresponds to Final step of DDPM samples

        # now t_steps basically has different noise levels with which we have to denoise
        # Main sampling loop.
        # here x is pure gaussian noise
        x_next = x.to(torch.float64) * t_steps[0]
        # map time step transitions [(0,1)......(16,17)]
        # here time represents noise
        for i, (sigma_t, sigma_t_next) in enumerate (zip(t_steps[:-1], t_steps[1:])):
            # for first iteration x_current is N(0, 80I) (80 is sigma_max)
            x_current = x_next
            # increase noise temporarily
            # we have deterministic sampling with current setting
            gamma = ( # basically sigma_t, or beta_t that controls stochastic noise injection
                min (self.edm_config.S_churn / num_steps, np.sqrt(2) - 1)
                if self.edm_config.S_min <= sigma_t <= self.edm_config.S_max else 0
            )

            # stochastic (langevin) exploration and ODE variance encapsulated in single variable
            sigma_t_hat = torch.as_tensor (sigma_t + gamma * sigma_t)

            # variance scaling factor of noise injection that decays as we approach 0
            # with current deterministic setting it will always be 0

            # STEP 1 :inject controlled noise
            beta_t = (sigma_t_hat ** 2 - sigma_t ** 2).sqrt()
            x_t = (
                x_current +  beta_t*
                self.edm_config.S_noise *
                torch.randn_like (x_current) 
            )

            # STEP 2: follow score gradient
            
            # Euler step.
            # x_t is signal with decayed stochastic noise injection
            # no need to pass CFG again since we already preset it in the "Partial " statement
            x0_approx = self.model_forward_wrapper (
                x_t.to(torch.float32),
                sigma_t_hat.to (torch.float32),
                y,
                model_forward_fxn,
                mask_ratio=mask_ratio,
                **kwargs 
            )['sample'].to(torch.float64)

            # x0_approx is approximation of x0 from x_t
            # xt = x0 + sigma_t * eps 
            # eps = (xt - x0) / sigma_t
            eps_o1 = (x_t - x0_approx) / sigma_t_hat
            x_next = x_t - (sigma_t_hat - sigma_t_next) * eps_o1   
            
            # Note that we don't try to solve for x0 in a single step as : x0 = x_T - sigma_t * eps
            # as that would overshoot, and ignore the dynamics of different noise levels
            # we instead step in terms of dt
            # Multiplying the eps with sigma_t would take a full denoising step immediately, using dt ensures
            # smoother updates, following the ODE trajectory rather than making a sudden larger jump

            # Heun Step : 2nd order correction
            if i < num_steps - 1:
                x0_approx = self.model_forward_wrapper (
                    x_next.to(torch.float32),
                    sigma_t_next.to(torch.float32),
                    y,
                    model_forward_fxn,
                    mask_ratio= mask_ratio,
                    **kwargs
                )['sample'].to(torch.float64)
                eps_o2 = (x_next - x0_approx) / sigma_t_next
                x_next = x_t - (sigma_t_hat - sigma_t_next) * (0.5 * eps_o1 + 0.5 * eps_o2)
        return x_next.to(torch.float32)
    
    @torch.no_grad()
    def generate (
        self,
        latent_prompt : torch.Tensor = None,
        prompt : Optional[list] = None,
        tokenized_prompts : Optional[torch.LongTensor] = None,
        attention_mask : Optional[torch.LongTensor] = None,
        guidance_scale : Optional[float] = 5.0,
        num_inference_steps: Optional[int] = 30,
        seed: Optional[int] = None,
        return_only_latents : Optional[bool] = False,
        **kwargs
    )-> torch.Tensor:
        # check caption prompt
        assert prompt is not None or tokenized_prompts is not None or latent_prompt is not None, f"Must provide either prompt or tokenized prompts or latent_prompt"
        device = self.vae.device # id model device during training
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)
        
        if latent_prompt is None:
            # caption prompt -> tokenize -> clip embed
            if tokenized_prompts is None:
                out = self.tokenizer.tokenize (prompt)
                tokenized_prompts = out['caption_idx']  # (B, 77)
                attention_mask = (
                    out['attention_mask'] if 'attention_mask' in out else None
                )
            
            text_embeddings = self.text_encoder.encode (
                tokenized_prompts.to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None
            )[0] # extract latent embeddings from returned tuple ((B,1,T,C), NONE)
        else:
            text_embeddings = latent_prompt

        # xT
        latents = torch.randn (
            (len(text_embeddings), self.dit.in_channels, self.latent_res, self.latent_res),
            device = device,
            generator=rng_generator
        )

        # iteratively denoise latents
        latents = self.edm_sampler_loop (
            latents,
            text_embeddings,
            num_inference_steps,
            cfg=guidance_scale
        )

        if return_only_latents:
            return latents

        # decode latents with VAE
        # scale back up with VAE scaling facotr
        latents = (1 / self.latent_scale) * latents
        torch_dtype = DATA_TYPES[self.dtype]
        image = self.vae.decode (latents.to(torch_dtype)).sample
        image = (image/2 + 0.5).clamp(0,1)
        image = image.float().detach()
        return image
    

def create_latent_diffusion (
        vae_name: str = None, #'stabilityai/stable-diffusion-xl-base-1.0',
        text_encoder_name: str = None, #'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',
        dit_arch: str = "MicroDiT_XL",
        latent_res: int = 32,
        in_channels: int = 4,
        pos_interp_scale: float = 1.0,
        dtype: str = 'bfloat16',
        latents_precomputed : bool = True,
        p_mean: float = -0.6,
        p_std: float = 1.2,
        train_mask_ratio: float = 0.0,
)-> LatentDiffusion:
    # Retrieve max sequence length (s) and text n_embd from text encoder
    if text_encoder_name is not None:
        seq_length, n_embd = get_text_encoder_embedding_format(text_encoder_name)
    else:
        # precomputed latents
        seq_length, n_embd = 77, 1024

    # TODO: change model_zoo and ['sample'] keys
    dit = getattr (model_zoo, dit_arch) (  # get class of specified dit_arch from the imported file
        input_res = latent_res,
        caption_n_embd = n_embd,
        pos_interp_scale = pos_interp_scale,
        in_channels=in_channels
    )
    
    
    if vae_name is not None:
        vae = AutoencoderKL.from_pretrained (vae_name, subfolder= None if vae_name=='ostris/vae-kl-f8-d16' else 'vae', torch_dtype=DATA_TYPES[dtype], pretrained=True)
    else:
        vae = None
    if text_encoder_name is not None:
        tokenizer = UniversalTokenizer (text_encoder_name)
        text_encoder = UniversalTextEncoder (text_encoder_name, weights_dtype=dtype, pretrained=True)
    else:
        tokenizer = None
        text_encoder = None
    
    assert (vae is not None)
    diffusion_model = LatentDiffusion (
        dit = dit,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer=tokenizer,
        latents_precomputed= latents_precomputed,
        dtype=dtype,
        latent_res=latent_res,
        p_mean=p_mean,
        p_std=p_std,
        train_mask_ratio=train_mask_ratio  
    )
    return diffusion_model


        