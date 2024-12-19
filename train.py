import torch
from torch import nn, optim

import ddp_trainer
from ddp_trainer import Trainer
from ddp_trainer.utils import evaluate

import dataset
import models

if __name__ == "__main__":
    ddp_trainer.init()

    # setup hyperparameters
    batch_size = 64
    max_length = 128
    learning_rate = 1e-5
    epochs = 10

    # load model and tokenizer
    model, tokenizer = models.load_model()

    # load datasets
    data_path = "./data/email_classification.csv"
    enron_path = "./data/enron_spam_data.csv"

    feature_cols = ["Message", "Subject"]
    train_set, val_set = dataset.load_enron_data(enron_path, feature_cols=feature_cols)

    # Modified collate for flexible features of enron spam dataset
    def collate_fn(batch):
        messages = [item[0][0] for item in batch]
        subjects = [item[0][1] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        
        # Encode both message and subject
        message_inputs = tokenizer(
            messages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        subject_inputs = tokenizer(
            subjects,
            padding=True,
            truncation=True,
            max_length=max_length // 4,
            return_tensors="pt"
        )
        
        # Combine the inputs
        encoded_inputs = {
            'message_input_ids': message_inputs.input_ids,
            'message_attention_mask': message_inputs.attention_mask,
            'subject_input_ids': subject_inputs.input_ids,
            'subject_attention_mask': subject_inputs.attention_mask
        }
        
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
    trainer = Trainer(
        model,
        loss_fn,
        optimizer,
        eval_fn=evaluate.multi_class_accuracy,
    )

    # train the model
    trainer.train(epochs, train_loader, val_loader)
