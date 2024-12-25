import matplotlib.pyplot as plt
import torch
from torch import nn, optim

import ddp_trainer
from ddp_trainer import Trainer
from ddp_trainer.utils import evaluate

import dataset
import models


def plot_learning_curve(train_loss, val_loss, val_acc, output_path="curve.png"):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.plot(train_loss, label="Train Loss")
    ax1.plot(val_loss, label="Validation Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(val_acc, label="Validation Accuracy", color="green")
    ax2.tick_params(axis="y")
    fig.legend(loc="upper left")
    plt.title("Learning Curve")
    fig.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    ddp_trainer.init()

    # setup hyperparameters
    batch_size = 16
    max_length = 128
    learning_rate = 1e-5
    epochs = 10

    # load model and tokenizer
    model, tokenizer = models.load_model()

    # load datasets
    data_path = "./data/processed_data.csv"
    train_set, val_set = dataset.load_data(data_path)

    # create dataloader
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        encoded_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded_inputs, labels

    train_loader = ddp_trainer.to_ddp_loader(
        train_set, batch_size=batch_size, num_workers=4, collate_fn=collate_fn
    )
    val_loader = ddp_trainer.to_ddp_loader(
        val_set, batch_size=batch_size, num_workers=4, collate_fn=collate_fn
    )

    # config loss funciton and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.1)
    trainer = Trainer(
        model,
        loss_fn,
        optimizer,
        eval_fn=evaluate.multi_class_accuracy,
        scheduler=scheduler,
    )

    # train the model
    trainer.train(epochs, train_loader, val_loader)

    # save the model
    if ddp_trainer.rank == 0:
        torch.save(model.state_dict(), "output/bert_spam.pt")

        history = trainer.history
        train_loss = history.get_train_losses()
        val_loss = history.get_val_losses()
        acc = history.get_evaluations()
        plot_learning_curve(train_loss, val_loss, acc, "output/learning_curve.png")
