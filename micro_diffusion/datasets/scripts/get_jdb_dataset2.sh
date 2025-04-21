#!/bin/bash

# Get user input for data directory and dataset size

datadir=$1
dataset_size=$2 # small or all
num_proc=$3


# A. Download a small subset (~1%) of the dataset, if specified
if [ "$dataset_size" == "small" ]; then
    echo "Downloading ~1% of the dataset..."
    python micro_diffusion/datasets/prepare/jdb/download2.py --datadir $datadir --max_image_size 512 --min_image_size 256 --valid_ids 6 7 --num_proc ${num_proc}
# Or download the entire dataset, if specified
elif [ "$dataset_size" == "all" ]; then
    echo "Downloading the full dataset... with processes " 
    echo ${num_proc}
    python micro_diffusion/datasets/prepare/jdb/download2.py --datadir $datadir --max_image_size 512 --min_image_size 256 --num_proc ${num_proc}
else
    echo "Invalid dataset size option. Please use 'small' or 'all'."
    exit 1
fi

# B. Convert dataset to MDS format.
python micro_diffusion/datasets/prepare/jdb/batch_convert.py --images_dir "${datadir}/raw/train/imgs/" --captions_jsonl "${datadir}/raw/train/train_anno_realease_repath.jsonl" --local_mds_dir "${datadir}/mds/train/" --batch_size 200 --num_workers 8
#python micro_diffusion/datasets/prepare/jdb/convert.py --images_dir "${datadir}/raw/valid/imgs/" --captions_jsonl "${datadir}/raw/valid/valid_anno_repath.jsonl" --local_mds_dir "${datadir}/mds/valid/"