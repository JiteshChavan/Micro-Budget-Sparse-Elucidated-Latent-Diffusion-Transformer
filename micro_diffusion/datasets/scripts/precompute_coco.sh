datadir= $1
num_gpus= $2

# start with 32 and watch nvidia smi
batch_size=$3 # use batch size of 8 for <16GB GPU memory

# Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
accelerate launch --multi_gpu --num_processes $num_gpus micro_diffusion/datasets/prepare/coco/precompute.py --datadir "${datadir}/mds/" \
    --savedir "${datadir}/coco_mds_latents/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size