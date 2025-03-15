num_gpus=$1
datadir=$2
batch_size=$3 # use batch size of 8 for <16GB GPU memory

# Local excecution
# bash micro_diffusion/datasets/scripts/precompute_jdb.sh 1 ./datadir/jdb/journeyDB 4
# C. Precompute latents across multiple GPUs. --multi_gpu 
python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
for split in train valid; do
    accelerate launch --num_processes $num_gpus ex_micro_diffusion/datasets/prepare/jdb/precompute.py --datadir "${datadir}/mds/$split/" \
        --savedir "${datadir}/jdb_mds_latents/$split/" --vae stabilityai/stable-diffusion-xl-base-1.0 \
        --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size
done