import torch
from torch.utils.data import DataLoader

import dataset
import models

if __name__ == "__main__":
    device = "cuda"

    # load datasets
    data_path = "./data/email_classification.csv"
    train_set, val_set = dataset.load_data(data_path)
    print("Training   samples:", len(train_set))
    print("Validation samples:", len(val_set))

    # load model
    model, tokenizer = models.load_model()
    model = model.to(device)

    MAX_LENGTH = 128
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        encoded_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return encoded_inputs, labels

    # create dataloader
    val_loader = DataLoader(val_set, 4, shuffle=True, collate_fn=collate_fn)
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
        print(outputs.shape)
