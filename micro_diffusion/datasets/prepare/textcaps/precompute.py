import torch.distributed as dist

if dist.is_initialized():
    dist.destroy_process_group()


import os
import time
from argparse import ArgumentParser
from typing import List, Optional

import numpy as np

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm

from micro_diffusion.datasets.prepare.textcaps.base import build_streaming_textcaps_precompute_dataloader
from micro_diffusion.models.utils import UniversalTextEncoder, DATA_TYPES

import gc

"""Example usage:
    CUDA_LAUNCH_BLOCKING=1 accelerate launch precompute.py \
    --datadir ./mds/ \
    --savedir ./mds_latents_sdxl1_dfnclipH14/ \
    --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --batch_size 32

    for windows
    accelerate launch precompute.py --datadir ./mds/ --savedir ./mds_latents_sdxl1_dfnclipH14/ --vae stabilityai/stable-diffusion-xl-base-1.0 --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size 4


"""

def extract_commandline_flags ():
    """Parse command-line arguments"""
    parser = ArgumentParser()

    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="Source path to load mds shards from (image, text pairs in native form)"
    )
    parser.add_argument (
        "--savedir",
        type=str,
        required=True,
        help="Destination path to store latent mds shards to"
    )
    parser.add_argument(
        "--vae",
        type = str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Name of VAE model to use for vision encoding.",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
        help="Name of model to use for text encoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device to use for encoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for random number generation.",
    )
    parser.add_argument(
        "--image_resolutions",
        type=int,
        nargs="+",
        default=[256, 512],
        help="List of image resolutions to use for processing."
    )
    parser.add_argument(
        # TODO: Introspect later
        "--save_images",
        default=False,
        action="store_true",
        help="If true saves images else only latents"
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Data type for the encoding models"
    )
    parser.add_argument (
        "--save_dtype",
        type=str,
        choices=("float16", "float32"),
        default="float16",
        help="Data type to save the latents"
    )

    args = parser.parse_args()
    if isinstance(args.image_resolutions, int):
        args.image_resolutions = [args.image_resolutions]
    return args

def main (args):
    """Precompute image and text latents and store them in MDS format.
    
    By default, we only save the image latents for 256x256 and 512x512 image resolutions using centre crop.

    Note that the image latents will be scaled by the vae_scaling_factor.
    """
    caption_key = "caption" # hardcode the image caption key to 'caption' in MDS dataset

    accelerator = Accelerator()
    device = accelerator.device
    device_idx = int(accelerator.process_index)
    # TODO: introspect later
   
    device = f"{device}"
    torch.manual_seed(args.seed + device_idx)
    torch.cuda.manual_seed (args.seed + device_idx)
    np.random.seed (args.seed + device_idx)

    dataloader = build_streaming_textcaps_precompute_dataloader (
        datadir=[args.datadir],
        batch_size=args.batch_size,
        resize_sizes=args.image_resolutions,
        drop_last=False,
        shuffle=False,
        caption_key=caption_key,
        tokenizer_name=args.text_encoder,
        prefetch_factor=2,
        num_workers = 2,
        persistent_workers=True,
        pin_memory=True
    )


    print (f"Device: {device_idx}, Dataloader sample count : {len(dataloader.dataset)}")
    print (f"MP Variable -> world size: {os.environ['WORLD_SIZE']} \
           RANK: {os.environ['RANK']}, {device}")
    
    vae = AutoencoderKL.from_pretrained (args.vae, subfolder="vae", torch_dtype=DATA_TYPES[args.model_dtype])
    print ("Initialized VAE:", args.vae)
    assert isinstance(vae, AutoencoderKL)


    vae = vae.to(device)
    gc.collect()

    text_encoder = UniversalTextEncoder(args.text_encoder, weights_dtype=args.model_dtype, pretrained=True)
    print ("Initialized text encoder:", args.text_encoder)
    text_encoder= text_encoder.to(device)
    gc.collect()

    columns = {
        caption_key : "str", # should be tokens
        f"{caption_key}_latents" : "bytes",
        "latents_256" : "bytes",
        "latents_512" : "bytes"
    }

    if args.save_images:
        columns["jpg"] = "jpeg"
    
    destination_path = os.path.join (args.savedir, str(accelerator.process_index))
    # TODO: change workers before sending to lambda
    writer = MDSWriter (out=destination_path, columns=columns, compression=None, size_limit=256 * (2**20), max_workers=16)

    for batch in tqdm (dataloader):
        image_256 = torch.stack(batch["image_0"]).to(device)
        image_512 = torch.stack(batch["image_1"]).to(device)

        # captions now has tokenized captions using the specified tokenizer, not strings
        # torch.stack 512 x (1, 77) -> (512, 1, 77)
        captions = torch.stack(batch[caption_key]).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=DATA_TYPES[args.model_dtype]):
                # vae returns a dict of type AutoencoderKLOuput, which contains latent distribution
                # accessed as ["latent_dist"] gives u sigma tensor, from which we sample (B, 4, 32, 32) latents
                # further the vae pipeline is supposed to scale the latents by the scaling factor (first batch std dev of pretraining data)
                # before processing the latents with diffusion pipeline, so we do the appropriate scaling
                # to "maintain unit variance"
                latent_dist_256 = vae.encode(image_256)
                assert isinstance (latent_dist_256, AutoencoderKLOutput)

                # tensor (B, 4, 32, 32)
                latents_256 = (latent_dist_256["latent_dist"].sample().data * vae.config.scaling_factor)
                latents_256 = latents_256.to(DATA_TYPES[args.save_dtype])

                latent_dist_512 = vae.encode(image_512)
                assert isinstance (latent_dist_512, AutoencoderKLOutput)

                latents_512 = (latent_dist_512["latent_dist"].sample().data * vae.config.scaling_factor)
                latents_512 = latents_512.to(DATA_TYPES[args.save_dtype])

                attention_mask = None
                if f"{caption_key}_attention_mask" in batch: # in case of t5 encoder
                    attention_mask = torch.stack(batch[f"{caption_key}_attention_mask"]).to(device)
                
                conditioning = text_encoder.encode (
                    # captions are stacked for a batch (512, 1, 77)
                    # encoder expects (B, 77) 2d input, reshape accordingly
                    # we only stack captions to compute latent shards
                    captions.view(-1, captions.shape[-1]),
                    attention_mask=attention_mask
                )[0].to(DATA_TYPES[args.save_dtype])

                # conditioning is (B, 1, 77, 1024)
        
        try:
            if isinstance(latents_256, torch.Tensor) and isinstance (latents_512, torch.Tensor):
                latents_256 = latents_256.detach().cpu().numpy()
                latents_512 = latents_512.detach().cpu().numpy()
            else:
                continue

            if isinstance(conditioning, torch.Tensor):
                conditioning = conditioning.detach().cpu().numpy()
            else:
                continue

            # write the batch to the MDS file
            for i in range (latents_256.shape[0]):
                mds_sample = {
                    # get the batch from og mds shards by indexing into "sample"
                    # (refers to getitem of super class which is streaming dataset)
                    # index into ith image of that sample batch from og mds shards
                    # get the textual caption from og mds shard
                    # thats what is happening in this line
                    caption_key : batch["sample"][i][caption_key],
                    f"{caption_key}_latents" : np.reshape(conditioning[i], -1).tobytes(), # (1, 77, 1024) stored as stream of bytes, text latents for one image, can be rearranged later
                    "latents_256" : latents_256[i].tobytes(), # (4, 32, 32)
                    "latents_512" : latents_512[i].tobytes(), # (4, 32, 32)
                }
                if args.save_images:
                    mds_sample["jpg"] = batch["sample"][i]["jpg"] # og image from og batch from og mds shard
                writer.write(mds_sample)
        except RuntimeError:
            print ("Runtime error CUDA, batch skipped")
    
    writer.finish()

    # wait for all process to finish
    accelerator.wait_for_everyone ()
    print (f"Process {accelerator.process_index} finished")
    time.sleep(10)

    # Merge the mds shards created by each device (only do on main process)
    if accelerator.is_main_process:
        shards_metadata = [
            os.path.join (args.savedir, str(i), "index.json")
            for i in range(accelerator.num_processes)
        ]
        merge_index (shards_metadata, out=args.savedir, keep_local=True)

if __name__ == "__main__":
    main(extract_commandline_flags())




