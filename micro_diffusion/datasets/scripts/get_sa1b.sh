#!/bin/bash
echo datadir ./datadir/sa1b
echo dataset_size_small/all
echo num_proc
# Get user input for data directory and dataset size
datadir=$1
dataset_size=$2 # small or all
num_proc=$3

# A. Download a small subset (~1%) of the dataset, if specified
if [ "$dataset_size" == "small" ]; then
    echo "Downloading 1% of the dataset..."
    python micro_diffusion/datasets/prepare/sa1b/download.py --datadir "${datadir}" --max_image_size 512 \
        --min_image_size 256 --data_fraction 0.01 --skip_existing  --num_proc $num_proc

# Or download the entire dataset, if specified
elif [ "$dataset_size" == "all" ]; then
    echo "Downloading the full dataset..."
    python micro_diffusion/datasets/prepare/sa1b/download.py --datadir "${datadir}" --max_image_size 512 \
        --min_image_size 256 --skip_existing  --num_proc $num_proc
else
    echo "Invalid dataset size option. Please use 'small' or 'all'."
    exit 1
fi

# B. Convert dataset to MDS format.
python  micro_diffusion/datasets/prepare/sa1b/convert.py --images_dir "${datadir}/raw/" --captions_dir "${datadir}/captions/" \
    --local_mds_dir "${datadir}/mds/" --num_proc $num_proc