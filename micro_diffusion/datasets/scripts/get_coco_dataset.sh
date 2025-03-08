#!/bin/bash

# Get user input for data directory and dataset size
datadir=$1




# COCO validation is fairly small. Download and shardify data in single script convert.py
python micro_diffusion/datasets/prepare/coco/convert.py --datadir "${datadir}/raw/" --local_mds_dir "${datadir}/mds/"