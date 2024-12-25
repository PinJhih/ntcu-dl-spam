import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.encoder = bert_model

    def forward(self, inputs):
        inputs = {k: v for k, v in inputs.items()}
        output = self.encoder(**inputs).logits
        return output

    def state_dict(self):
        return self.encoder.state_dict()


def load_model(model_name="google-bert/bert-large-uncased"):
    # load pretrained bert model
    bert = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # create BertClassifier
    model = BertClassifier(bert)
    return model, tokenizer
