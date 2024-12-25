import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

import models
import dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# load trained model
model, tokenizer = models.load_model()
state_dict = torch.load("output/bert_spam.pth", weights_only=True)
model.load_state_dict(state_dict)
model.to(device)


# load dataset
def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return encoded_inputs, labels


data_path = "./data/processed_data.csv"
train_set, val_set = dataset.load_data(data_path)
val_loader = DataLoader(val_set, batch_size=128, num_workers=4, collate_fn=collate_fn)

# inference
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x, y in tqdm(val_loader, desc="Inference"):
        inputs, labels = x.to(device), y.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute F1 score
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"F1 Score: {f1}")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"],
)
plt.title(f"Confusion Matrix - F1: {f1:.4f}")
plt.savefig("output/confusion_matrix.png")
