# Sparse Elucidated Latent Diffusion Transformer for text guided image synthesis
This repository contains the implementation resources for a **1.2B** parameters Sparse Ellucidated Latent Diffusion Transformer Model, achieving state-of-the-art image generation results, achieving an FID of 12.7 in zero-shot generation on the COCO dataset, at a significantly reduced computational cost of **$1314**. 

# Results:
![](./assets/owl.png)
Prompt: *"Owl with moon as backdrop in _____ style"*
Styles: *Pixel-art, cyberpunk, starry night*"
---
![](./assets/lotuspond.png)
Prompt: *"Moonlit night over a pond of lotuses and floating candles in drawn in the style of Starry Night by Vincent Van Gogh"*
---
![](./assets/knight.png)
Prompt: *"concept art of a lady knight in ____ style"*
Styles: *Pixel-art, cyberpunk, starry night*"
---
![](./assets/dragon.png)
Prompt: *"A celestial dragon soaring through nebula clouds, sparkling with stardust"*
---
![](./assets/mordor.png)
Prompt: *"In the land of mordor, where shadows lie"*
---
![](./assets/mage.png)
Prompt: *"A cosmic mage conjuring constellations in a swirling vortex of stars"*
---
![](./assets/misc.png)

# Methodology:










attained State of the art results at micro budget, primarily achieved by pretraining with patch masking ratio of 75% to reduce input seq length to diffusion transformer backbone and utilizing a lightweight patchmixer transformer to capture global context prior to applying masking to retain semantic information of entire image and increase the effectiveness of pretraining task to learn/distill representation for masked denoising process. The learned representation is further refined after pretraining stage by fine tuning with 0 masking ratio.

First we pretrain the model on 256x256 images then fine tune on the same resolution, and repeat the pretraining into fine tuning pipleline on 512x512 images by scaling the sinusoidal positional embeddings appropriately.



The model learns to basically generates images from text prompts using state of the art diffusion modeling has 1.2B parameters Expert Choice style Mixture of experts, diffusion transformer backbone, runs cross attention on text latent embeddings from Open clip model (openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378) and uses Variational autoencoder (model:stabilityai/stable-diffusion-xl-base-1.0) to translate images into latent space to make training computationally feasible and facilitate high resolution image synthesis exploiting latent diffusion.



it was trained on 25M image-text pairs from JourneyDB, diffusionDB, SegmentAnything1B (sa1b) datasets, using Fully Sharded Data Parallel and Hydracomposer framework
