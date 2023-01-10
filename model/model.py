from transformers import (BertTokenizer, BertModel)
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityModel(nn.Module):
    def __init__(self, bert_dim, l1_size, output_size):
        super(SimilarityModel, self).__init__()
        self.bert_dim = 768
        self.l1_size = 54
        self.output_size = 20
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.similarity_model = SentenceTransformer('all-distilroberta-v1')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert_dim + self.l1_size, self.output_size)

        for param in self.similarity_model.parameters():
            param.requires_grad = False

        l1_labels = pd.read_csv("data/level1-labels-training.tsv", delimiter="\t").columns.to_numpy()[1:]
        self.l1_embeds = []
        for label in l1_labels:
            self.l1_embeds.append(self.similarity_model.encode(label))

    def cos_similarity(self, premise: np.ndarray, label: np.ndarray):
        return np.dot(premise, label.T) / (np.linalg.norm(premise) * np.linalg.norm(label))

    def forward(self, premise):
        tokenized_premise = self.tokenizer.encode(premise, return_tensors='pt').to(device)
        bert_output = self.bert(tokenized_premise)["pooler_output"]

        sentence_bert_premise = self.similarity_model.encode(premise)
        array = np.zeros((1, self.l1_size))
        for idx, embed in enumerate(self.l1_embeds):
            sentence_bert_l1 = embed
            similarity = self.cos_similarity(sentence_bert_premise, sentence_bert_l1)
            array[0, idx] = similarity

        concat = torch.concat([bert_output, torch.tensor(array, dtype=torch.float32).to(device)], dim=1)
        output = self.linear(self.dropout(concat))

        return output
