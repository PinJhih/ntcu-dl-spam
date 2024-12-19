import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


class SpamDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # convert label to 0/1
        df["label"] = df["label"].map({"ham": 0, "spam": 1})

        self.x = df["email"].values
        self.y = torch.tensor(df["label"].values, dtype=torch.int64)
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

class EnronSpamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols=None):
        # convert label to 0/1
        df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})

        self.messages = df["Message"].values
        self.subjects = df["Subject"].values
        self.y = torch.tensor(df["label"].values, dtype=torch.int64)
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        message = self.messages[idx]
        subject = self.subjects[idx]
        y = self.y[idx]
        return (message, subject), y


def load_data(csv_path, val_ratio=0.25, random_seed=42):
    # Set random seeds for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    
    # read csv file and create dataset object
    data = pd.read_csv(csv_path)
    dataset = SpamDataset(data)

    # split train/val set
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset

def load_enron_data(csv_path, val_ratio=0.25, random_seed=42):
    generator = torch.Generator().manual_seed(random_seed)
    
    # Using EnronSpamDataset class
    data = pd.read_csv(csv_path)
    dataset = EnronSpamDataset(data)

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset
