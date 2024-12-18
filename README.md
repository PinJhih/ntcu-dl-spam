# NTCU-DL 期末報告 - Spam Detection

## Dataset
- [Email Classification (Ham-Spam)](https://www.kaggle.com/datasets/prishasawhney/email-classification-ham-spam)
- [enron_spam_data](https://github.com/MWiechmann/enron_spam_data)

## Setup

Install Pytorch: https://pytorch.org/get-started/locally/

```bash
git clone https://github.com/PinJhih/ntcu-dl-spam.git
cd ntcu-dl-spam && pip install -r requirements

# install DDP Trainer
mkdir -p ./lib && cd ./lib 
git clone https://github.com/PinJhih/ddp-trainer.git && cd ddp-trainer
pip install -e .
```
