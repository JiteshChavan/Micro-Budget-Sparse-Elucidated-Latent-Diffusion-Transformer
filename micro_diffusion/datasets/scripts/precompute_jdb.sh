echo num_gpus
echo datadir
echo batch_size

num_gpus=$1
datadir=$2
batch_size=$3 # use batch size of 8 for <16GB GPU memory

# C. Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
for split in train valid; do
    accelerate launch --multi_gpu --num_processes $num_gpus micro_diffusion/datasets/prepare/jdb/precompute.py --datadir "${datadir}/mds/$split/" \
        --savedir "${datadir}/jdb_mds_latents/$split/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
        --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size
done