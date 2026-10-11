"""
Downloads the official Kaggle 'Individual household electric power consumption' dataset.
"""
import os
import shutil
import kagglehub


def download_dataset():
    print("Downloading 'uciml/electric-power-consumption' from Kaggle...")
    path = kagglehub.dataset_download("uciml/electric-power-consumption")
    print(f"Downloaded to: {path}")

    files = os.listdir(path)
    print(f"Files found: {files}")

    dest_dir = "dataset"
    os.makedirs(dest_dir, exist_ok=True)

    for f in files:
        src = os.path.join(path, f)
        dst = os.path.join(dest_dir, f)
        if not os.path.exists(dst):
            print(f"Copying {f} to {dest_dir}...")
            shutil.copy(src, dst)
            print("Copy complete.")
        else:
            print(f"{f} already exists in {dest_dir}.")


if __name__ == "__main__":
    download_dataset()
