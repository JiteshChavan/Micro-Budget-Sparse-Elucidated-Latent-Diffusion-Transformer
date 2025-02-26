from micro_diffusion.models.utils import UniversalTextEncoder
from accelerate import Accelerator

import torch
from micro_diffusion.models.utils import UniversalTokenizer

"""accelerate launch --num_processes 1 clipplay.py"""

acc = Accelerator()
device = acc.device
device_idx = int(acc.process_index)

print ("!!!!!!! downloading encoder!!!!!")
text_encoder = UniversalTextEncoder("openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378", weights_dtype="bfloat16", pretrained=True)
print ("!!!!!!! Done!!!!!")
import gc
text_encoder.to(device)
gc.collect()
caption = ["pep my goat", "antony best", "test"]
tokenizer = UniversalTokenizer ('openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378')



a = tokenizer.tokenize(caption)['caption_idx']
a = a.to(device)
print (type(a), a.shape)

a, b = text_encoder.encode (a)
print (type(a), a.shape, type(b))