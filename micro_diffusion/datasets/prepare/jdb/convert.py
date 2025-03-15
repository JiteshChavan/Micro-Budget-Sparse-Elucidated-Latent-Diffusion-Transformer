import os
import json
from glob import glob
from argparse import ArgumentParser
from PIL import Image
from streaming.base import MDSWriter
from tqdm import tqdm

'''
Example usage:
python convert.py --images_dir ./journeyDB/raw/train/imgs/ \
    --captions_jsonl ./journeyDB/raw/train/train_anno_realease_repath.jsonl \
    --local_mds_dir ./journeyDB/mds/train/
'''


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Path to local dir with all images',
    )
    parser.add_argument(
        '--captions_jsonl',
        type=str,
        required=True,
        help='Path to jsonl file with all captions',
    )
    parser.add_argument(
        '--local_mds_dir',
        type=str,
        default='',
        help='Directory to store mds shards.',
    )
    return parser.parse_args()


def convert_to_mds(args: ArgumentParser):
    """Converts JourneyDB dataset to mds format."""
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
    
    print ("creating archive list")
    # Retrieving achieve indies, in case only a subset of the data is downloaded
    valid_archive_idx = [
        os.path.basename(p) for p in glob(os.path.join(args.images_dir, '*'))
    ]
    print (valid_archive_idx)
    print ("done")
    
    
    print ("opened captions file")
    with open (args.captions_jsonl, 'r') as f:
        for line in tqdm(f): # stream file line by line
            try:
                d = json.loads(line)
                cap, p = d['prompt'], d['img_path'].strip('./')
                if os.path.dirname(p) not in valid_archive_idx:
                    continue
        
                img = Image.open(os.path.join(args.images_dir, p))
                w, h = img.size
                mds_sample = {
                    'jpg': img,
                    'caption': cap,
                    'width': w,
                    'height': h,
                }
                writer.write(mds_sample)
            except Exception as e:
                print(
                    "Something went wrong in reading caption, "
                    f"skipping writing this sample in mds. Error: {e}"
                )

    writer.finish()


def main():
    print ("start")
    args = parse_arguments()
    convert_to_mds(args)


if __name__ == '__main__':
    main()