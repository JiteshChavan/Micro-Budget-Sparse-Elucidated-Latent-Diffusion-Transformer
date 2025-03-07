import torch
from micro_diffusion.models.utils import UniversalTokenizer, UniversalTextEncoder

captions = [
      "Nature in silhouette of an owl",
      "cyberpunk panda ",
      "majestic knight in shining armor",
      "waterfall dragon",
      "Robot in forest",
      "Shield maiden with wings",
      "a shadowy land of ghosts and specters",
      "woman with red cape long hair",
      "galaxy volcano explosion",
      "sage crystal orb",
]


tokenizer = UniversalTokenizer ('openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378')
clip = UniversalTextEncoder ("openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378", weights_dtype="bfloat16", pretrained=True).to('cuda')
latents = (clip.encode(tokenizer.tokenize(captions)['caption_idx'].to('cuda'))[0])
latents.shape
save_dir = "./caption_latents.pt"
torch.save(latents.cpu(), save_dir)
