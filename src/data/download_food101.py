"""
Food-101 Dataset Downloader & Verifier
Source: ETH Zürich Computer Vision Lab
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Custom progress bar for urllib download tracking."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_food101(destination_dir: str = "data/raw"):
    """
    Downloads the official Food-101 archive (~5GB) and extracts it.
    If the archive or extracted folder already exists, skips downloading.
    """
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    dataset_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    tar_path = dest_path / "food-101.tar.gz"
    extracted_folder = dest_path / "food-101"

    if extracted_folder.exists() and (extracted_folder / "images").exists():
        print(f"[INFO] Food-101 dataset already extracted at: {extracted_folder.resolve()}")
        return extracted_folder

    if not tar_path.exists():
        print(f"[INFO] Downloading Food-101 (~5GB) from {dataset_url} ...")
        print("[NOTE] On Google Colab, this takes ~1-2 minutes over cloud connection.")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="food-101.tar.gz") as t:
            urllib.request.urlretrieve(dataset_url, filename=tar_path, reporthook=t.update_to)
        print("[INFO] Download completed successfully.")
    else:
        print(f"[INFO] Archive {tar_path} already exists. Skipping download.")

    print(f"[INFO] Extracting archive to {dest_path.resolve()} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_path)
    print("[INFO] Extraction completed.")

    return extracted_folder


if __name__ == "__main__":
    download_and_extract_food101()
