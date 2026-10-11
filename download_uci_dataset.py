"""
Download the raw Individual Household Electric Power Consumption dataset from UCI / mirror.
"""
import urllib.request
import zipfile
import os

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
OUT_ZIP = "dataset/household_power.zip"

def try_download():
    os.makedirs("dataset", exist_ok=True)
    print(f"Connecting to {URL}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(URL, OUT_ZIP)
    print("Download finished. Extracting...")
    with zipfile.ZipFile(OUT_ZIP, 'r') as zip_ref:
        zip_ref.extractall("dataset")
    print("Extracted successfully!")

if __name__ == "__main__":
    try_download()
