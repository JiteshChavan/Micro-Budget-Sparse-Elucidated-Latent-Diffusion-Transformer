import torch
from micro_diffusion.models.utils import UniversalTokenizer, UniversalTextEncoder

captions = [
      "portrait of a beautiful lady with purple hair. Green vines and leaves surrounding her",
      "Ocean wave, aesthetic, soothing, turquoise blue",
      "Funny portrait of a koala",
      "Mysterious tree surrounded by lake",
      "A heroic girl's silhouette with moon as backdrop, moon is partially covered by clouds, its evening",
      "An armed hero with cape standing near explosion, warm color scheme",
      "Malevolent wizard with his staff and hat",
      "Watercolor splash art of a butterfly, purple yellow green blue",
      "A bird on branch with moon backdrop, ominous, scary ",
      "Sunset over mountains, river of blood",
      "monster in ominous forest, lake, moon, night",
      "volcano galaxy explosion blue purple ominous",
      "Charizard using flamethrower, cinematic shot",
      "Harry Potter playing quidditch in the grounds of hogwarts castle",
]


tokenizer = UniversalTokenizer ('openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378')
clip = UniversalTextEncoder ("openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378", weights_dtype="bfloat16", pretrained=True).to('cuda')
print(sum(p.numel() for p in clip.encoder.clip_model.parameters()))


latents = (clip.encode(tokenizer.tokenize(captions)['caption_idx'].to('cuda'))[0])
latents.shape
save_dir = "./caption_latents.pt"
torch.save(latents.cpu(), save_dir)
