import os
import json
from glob import glob
from argparse import ArgumentParser
from PIL import Image
from streaming.base import MDSWriter
from tqdm import tqdm
import multiprocessing


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--captions_jsonl', type=str, required=True)
    parser.add_argument('--local_mds_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def batched_jsonl_reader(file_path, batch_size):
    batch = []
    with open(file_path, 'r') as f:
        for line in f:
            batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def process_sample(line, images_dir, valid_archive_idx):
    try:
        d = json.loads(line)
        cap, p = d['prompt'], d['img_path'].strip('./')
        archive_dir = os.path.dirname(p)
        if archive_dir not in valid_archive_idx:
            return None

        img_path = os.path.join(images_dir, p)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        return {
            'jpg': img,
            'caption': cap,
            'width': w,
            'height': h,
        }
    except Exception as e:
        print(f"[WARN] Skipping sample: {e}")
        return None


def worker_process(batch, images_dir, valid_archive_idx):
    buffer = []
    for line in batch:
        sample = process_sample(line, images_dir, valid_archive_idx)
        if sample is not None:
            buffer.append(sample)
    return buffer


def chunkify(lst, n):
    """Split lst into n chunks as evenly as possible."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def convert_to_mds(args):
    columns = {
        'width': 'int32',
        'height': 'int32',
        'jpg': 'jpeg',
        'caption': 'str',
    }

    writer = MDSWriter(
        out=args.local_mds_dir,
        columns=columns,
        compression=None,
        size_limit=2 * (2**30),
        max_workers=64,
    )

    valid_archive_idx = [
        os.path.basename(p) for p in glob(os.path.join(args.images_dir, '*'))
    ]

    total_lines = count_lines(args.captions_jsonl)
    print(f"Processing {total_lines} samples with {args.num_workers} workers...")

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for batch in tqdm(
            batched_jsonl_reader(args.captions_jsonl, args.batch_size),
            total=(total_lines + args.batch_size - 1) // args.batch_size,
            desc="Converting to MDS"
        ):
            # Split this batch into N sub-batches for N workers
            sub_batches = chunkify(batch, args.num_workers)
            results = pool.starmap(worker_process, [
                (sub, args.images_dir, valid_archive_idx) for sub in sub_batches
            ])

            buffer = []
            for sub_result in results:
                buffer.extend(sub_result)
                if len(buffer) >= 32:
                    writer.write(buffer)
                    buffer.clear()

            if buffer:
                writer.write(buffer)

    writer.finish()


def main():
    args = parse_arguments()
    convert_to_mds(args)


if __name__ == '__main__':
    main()
