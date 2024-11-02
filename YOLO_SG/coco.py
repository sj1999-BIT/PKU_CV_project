import os
import requests
from tqdm import tqdm
import zipfile


def download_file(url, filename):
    """
    Download file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def download_coco(root_dir):
    """
    Download COCO dataset
    """
    # Create directories
    os.makedirs(root_dir, exist_ok=True)

    # URLs for different parts of COCO dataset
    urls = {
        # 'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
        # 'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
        'test2017.zip': 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations_trainval2017.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }

    # Download and extract each file
    for filename, url in urls.items():
        filepath = os.path.join(root_dir, filename)

        # Download if file doesn't exist
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            download_file(url, filepath)

        # Extract if not already extracted
        extract_dir = os.path.join(root_dir, filename.split('.')[0])
        if not os.path.exists(extract_dir):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(root_dir)

        print(f"Completed {filename}")


if __name__ == "__main__":
    # Specify your download directory
    download_dir = "./coco_dataset"
    download_coco(download_dir)