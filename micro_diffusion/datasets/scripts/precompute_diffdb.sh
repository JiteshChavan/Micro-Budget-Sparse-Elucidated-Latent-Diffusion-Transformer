
echo num_gpus
echo datadir ./datadir/diffdb
echo batch_size 



num_gpus=$1
datadir=$2
batch_size=$3



# Precompute latents across multiple GPUs.
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
accelerate launch --multi_gpu --num_processes $num_gpus micro_diffusion/datasets/prepare/diffdb/precompute.py --datadir "${datadir}/mds/" \
    --savedir "${datadir}/diffdb_mds_latents/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size