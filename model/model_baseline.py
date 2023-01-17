from transformers import (BertTokenizer, BertModel)
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityModel(nn.Module):
    def __init__(self, output_size, l1_labels, l1_to_l2_map):
        super(SimilarityModel, self).__init__()
        self.bert_dim = 768
        self.output_size = output_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert_dim, self.output_size)

    def forward(self, premise):
        tokenized_premise = self.tokenizer.encode(premise, return_tensors='pt').to(device)
        bert_output = self.bert(tokenized_premise)["pooler_output"]

        output = self.linear(self.dropout(bert_output))

        return output
