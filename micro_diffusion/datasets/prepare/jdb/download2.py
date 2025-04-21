import os
import tarfile
import argparse
import shutil
from glob import iglob
from huggingface_hub import hf_hub_download
from multiprocessing.pool import ThreadPool
from PIL import Image, UnidentifiedImageError, ImageFile
from torchvision import transforms
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_arguments():
    parser = argparse.ArgumentParser(description='Optimized JourneyDB downloader.')
    parser.add_argument('--datadir', type=str, default='./journeyDB/')
    parser.add_argument('--max_image_size', type=int, default=512)
    parser.add_argument('--min_image_size', type=int, default=256)
    parser.add_argument('--valid_ids', type=int, nargs='+', default=list(np.arange(200)))
    parser.add_argument('--num_proc', type=int, default=8)
    return parser.parse_args()

def extract_tgz(file_path, extract_dir):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

def download_and_process_metadata(args):
    metadata_files = [
        ('data/train', 'train_anno.jsonl.tgz'),
        ('data/train', 'train_anno_realease_repath.jsonl.tgz'),
        ('data/valid', 'valid_anno_repath.jsonl.tgz'),
        ('data/test', 'test_questions.jsonl.tgz'),
        ('data/test', 'imgs.tgz'),
    ]

    for subfolder, filename in metadata_files:
        hf_hub_download(
            repo_id="JourneyDB/JourneyDB",
            repo_type="dataset",
            subfolder=subfolder,
            filename=filename,
            local_dir=args.datadir_compressed,
            local_dir_use_symlinks=False,
        )
        extract_tgz(
            os.path.join(args.datadir_compressed, subfolder, filename),
            os.path.join(args.datadir_compressed, subfolder)
        )

    shutil.copy(
        f'{args.datadir_compressed}/data/train/train_anno_realease_repath.jsonl',
        f'{args.datadir_raw}/train/train_anno_realease_repath.jsonl'
    )
    shutil.copy(
        f'{args.datadir_compressed}/data/valid/valid_anno_repath.jsonl',
        f'{args.datadir_raw}/valid/valid_anno_repath.jsonl'
    )
    shutil.copy(
        f'{args.datadir_compressed}/data/test/test_questions.jsonl',
        f'{args.datadir_raw}/test/test_questions.jsonl'
    )
    shutil.move(
        f'{args.datadir_compressed}/data/test/imgs',
        f'{args.datadir_raw}/test/'
    )

def resize_and_move_image(args, downsize, src_file, dst_dir):
    try:
        img = Image.open(src_file)
        w, h = img.size

        if min(w, h) < args.min_image_size:
            return

        if min(w, h) > args.max_image_size:
            img = downsize(img)

        img.save(os.path.join(dst_dir, os.path.basename(src_file)))
        os.remove(src_file)
    except (UnidentifiedImageError, OSError) as e:
        print(f"Error {e}, File: {src_file}")

def download_uncompress_resize(args, split, idx):
    hf_hub_download(
        repo_id="JourneyDB/JourneyDB",
        repo_type="dataset",
        subfolder=f"data/{split}/imgs",
        filename=f"{idx:03}.tgz",
        local_dir=args.datadir_compressed,
        local_dir_use_symlinks=False,
    )

    tgz_path = f'{args.datadir_compressed}/data/{split}/imgs/{idx:03}.tgz'
    extract_path = f'{args.datadir_compressed}/data/{split}/imgs/'
    extract_tgz(tgz_path, extract_path)
    os.remove(tgz_path)

    dst_dir = f'{args.datadir_raw}/{split}/imgs/{idx:03}'
    os.makedirs(dst_dir, exist_ok=True)

    downsize = transforms.Resize(
        args.max_image_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    )

    files = list(iglob(f'{extract_path}/{idx:03}/*'))
    with ThreadPool(processes=4) as pool:
        pool.starmap(
            resize_and_move_image,
            [(args, downsize, f, dst_dir) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )

def main():
    args = parse_arguments()

    os.makedirs(args.datadir, exist_ok=True)
    args.datadir_compressed = os.path.join(args.datadir, 'compressed')
    args.datadir_raw = os.path.join(args.datadir, 'raw')
    os.makedirs(args.datadir_compressed, exist_ok=True)
    os.makedirs(args.datadir_raw, exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, "train", "imgs"), exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, "valid", "imgs"), exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, "test"), exist_ok=True)

    download_and_process_metadata(args)

    from multiprocessing import Pool
    pool_args = [('train', i) for i in args.valid_ids] + [('valid', i) for i in args.valid_ids]
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(download_uncompress_resize, [(args, split, idx) for split, idx in pool_args])

if __name__ == "__main__":
    main()
