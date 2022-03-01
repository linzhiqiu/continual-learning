"""
Download the unlabeled images to root/all_images, seperated by bucket
"""
import argparse
from tqdm import tqdm
import time
import os
import json
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

MAX_NUM_OF_TRAILS = 3 # Max num to try downloading for each images
sleep_time = 1 # if download fail, sleep for one second and retry (until MAX_NUM_OF_TRAILS times)
MIN_IMAGE_SIZE = 10 # minimum byte size of the downloaded image

argparser = argparse.ArgumentParser()
argparser.add_argument("--root",
                       default='/data3/zhiqiul/CLEAR-10-PUBLIC',
                       help="The path to the downloaded folder with all_metadata.json")
argparser.add_argument("--max_workers",
                        type=int, default=128,
                        help="The number of parallel workers (threads) for image download.")

def load_json(json_location, default_obj=None):
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj

def save_as_json(json_location, obj, indent=None):
    with open(json_location, "w+") as f:
        json.dump(obj, f, indent=indent)

def download_image(img_path, url, MAX_NUM_OF_TRAILS=MAX_NUM_OF_TRAILS, sleep_time=sleep_time, MIN_IMAGE_SIZE=MIN_IMAGE_SIZE):
    """Return true if image is valid and successfully downloaded
    """
    trials = 0
    while True:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save(img_path)
            if os.path.getsize(img_path) < MIN_IMAGE_SIZE:
                # This is to make sure it is not an empty Flickr default image
                os.remove(img_path)
                return False
            return True
        except:
            if trials < MAX_NUM_OF_TRAILS:
                time.sleep(sleep_time)
                trials += 1
                continue
            else:
                return False


if __name__ == '__main__':
    args = argparser.parse_args()
    
    save_path = Path(args.root)
    
    labeled_images_path = save_path / 'labeled_images'
    labeled_metadata_path = save_path / 'labeled_metadata'
    assert labeled_metadata_path.exists(), f"{labeled_metadata_path} does not exist."
    assert labeled_images_path.exists(), f"{labeled_images_path} does not exist."
    
    all_images_path = save_path / 'all_images'
    all_metadata_path = save_path / 'all_metadata'
    all_metadata_json_path = save_path / 'all_metadata.json'
    assert all_metadata_json_path.exists(), f"{all_metadata_json_path} does not exist."
    assert all_metadata_path.exists(), f"{all_metadata_path} does not exist."
    
    all_metadata_dict = load_json(all_metadata_json_path)
    if not all_images_path.exists():
        all_images_path.mkdir()
    
    for b_idx in all_metadata_dict:
        all_metadata_path_i = all_metadata_path / (b_idx + ".json")
        all_metadata_i = load_json(all_metadata_path_i)
        all_images_path_i = all_images_path / b_idx
        all_images_path_i.mkdir(exist_ok=True)
        
        failed = {}
        success_count = 0
        d = {}
        start = time.time()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for ID in tqdm(all_metadata_i):
                meta = all_metadata_i[ID]
                img_path = str(save_path / meta['IMG_PATH'])
                url = meta['DOWNLOAD_URL']
                d[executor.submit(download_image, img_path, url)] = (img_path, url)
            
            print(f"There are {len(d.keys())} images to download for bucket {b_idx}")
            for future in as_completed(d):
                img_path, url = d[future]
                try:
                    success = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                    failed[url] = img_path
                else:
                    if success:
                        success_count += 1
                    else:
                        failed[url] = img_path
        end = time.time()

        print(f"{end-start} seconds for downloading bucket {b_idx}")
        total_runs = len(failed.keys()) + success_count
        if total_runs != len(d.keys()):
            print("Error in multiprocessing")
            exit(0)
        else:
            print(f"Among the {len(d.keys())} images in bucket {b_idx}, {len(failed.keys())} failed.")
            if len(failed.keys()) > 0:
                failed_path = all_images_path / "failed"
                if not failed_path.exists():
                    failed_path.mkdir()
                failed_path_i = failed_path / f"{b_idx}.json"
                print(f"Saving all failed urls and paths to {failed_path_i}")
                save_as_json(failed_path_i, failed, indent=4)      
    
    