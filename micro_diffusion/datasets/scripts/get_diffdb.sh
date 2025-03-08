#!/bin/bash

# Get user input for data directory and dataset size

echo datadir
echo dataset_size_small/all # small or all
echo num_proc

datadir=$1
dataset_size=$2 # small or all
num_proc=$3

# A. Download a small subset (~1%) of the dataset, if specified
if [ "$dataset_size" == "small" ]; then
    echo "Downloading ~1% of the dataset..."
    python micro_diffusion/datasets/prepare/diffdb/download.py --datadir $datadir --max_image_size 512 --min_image_size 256 \
        --valid_ids $(seq 1 140) --num_proc $num_proc

# Or download the entire dataset, if specified
elif [ "$dataset_size" == "all" ]; then
    echo "Downloading the full dataset..."
    python micro_diffusion/datasets/prepare/diffdb/download.py --datadir $datadir --max_image_size 512 --min_image_size 256 \
        --num_proc $num_proc
else
    echo "Invalid dataset size option. Please use 'small' or 'all'."
    exit 1
fi

# B. Convert dataset to MDS format.
python micro_diffusion/datasets/prepare/diffdb/convert.py --images_dir "${datadir}/raw/" --local_mds_dir "${datadir}/mds/" \
    --num_proc $num_proc --safety_threshold 0.2