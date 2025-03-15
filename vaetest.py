import torch
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder='vae',  # Change subfolder to appropriate one in hf_hub, if needed
        torch_dtype=torch.bfloat16,
    )

vae.to('cuda')

a = torch.randn(1, 3, 256, 256).to(dtype=torch.bfloat16)
a = a.to('cuda')

out = vae.encode(a)
latents_256 = (
                    out['latent_dist'].sample().data
                ).to(torch.bfloat16)
print (latents_256.shape, latents_256.dtype)