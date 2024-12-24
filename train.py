import matplotlib.pyplot as plt
import torch
from torch import nn, optim

import ddp_trainer
from ddp_trainer import Trainer
from ddp_trainer.utils import evaluate

import dataset
import models


def plot(path, train_loss, val_loss):
    plt.figure(10, 6)
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.savefig(path)


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

        history = trainer.history()
        plot("output/loss.png", history.get_train_losses(), history.get_val_losses())

        print(history.get_evaluations())
