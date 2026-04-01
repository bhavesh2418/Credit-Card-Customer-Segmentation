"""
Script: download_data.py
Purpose: Download Credit Card dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        print("ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env")
        sys.exit(1)

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "arjunbhasin2013/ccdata",
        path=str(DATA_RAW),
        unzip=True
    )

    files = list(DATA_RAW.glob("*.csv"))
    if files:
        for f in files:
            print(f"Downloaded: {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    else:
        print("Download may have completed — check data/raw/")


if __name__ == "__main__":
    download_dataset()
