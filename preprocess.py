import os
import requests
import zipfile

import pandas as pd

DATA_DIR = "./data"
RAW_DATA_PATH = f"{DATA_DIR}/enron_spam_data.csv"
ZIP_DATA_PATH = f"{DATA_DIR}/enron_spam_data.zip"
ZIP_DATA_URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/refs/heads/master/enron_spam_data.zip"
OUTPUT_PATH = f"{DATA_DIR}/processed_data.csv"


def check_data_dir():
    if os.path.exists(DATA_DIR):
        return
    os.makedirs(DATA_DIR)
    print(f"Created directory {DATA_DIR}")


def check_raw_data():
    check_data_dir()
    if os.path.exists(RAW_DATA_PATH):
        return

    # Download raw data (zip file)
    response = requests.get(ZIP_DATA_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to download raw data from {ZIP_DATA_URL}")
    with open(ZIP_DATA_PATH, "wb") as f:
        f.write(response.content)
        print(f"Downloaded raw data from {ZIP_DATA_URL}")

    # Extract raw data
    with zipfile.ZipFile(ZIP_DATA_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
        print(f"Extracted raw data to {DATA_DIR}")


def preprocess():
    check_raw_data()
    raw_data = pd.read_csv(RAW_DATA_PATH).fillna("").astype(str)

    subject = raw_data["Subject"]
    message = raw_data["Message"]
    email = subject + " " + message

    processed_data = pd.DataFrame(
        {
            "email": email,
            "label": raw_data["Spam/Ham"],
        }
    )
    processed_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessed data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()
