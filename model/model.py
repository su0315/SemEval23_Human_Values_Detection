from transformers import (BertTokenizer, BertModel)
from sentence_transformers import SentenceTransformer
from transformers import logging
import random
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
from tqdm import tqdm

l1_labels = pd.read_csv("../data/level1-labels-training.tsv", delimiter="\t").columns.to_numpy()[1:]


class SimilarityModel(nn.Module):
    def __init__(self, bert_dim, l1_size, output_size):
        super(SimilarityModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.similarity_model = SentenceTransformer('all-distilroberta-v1')
        self.bert_dim = 768
        self.l1_size = 54
        self.output_size = 20
        self.linear = nn.Linear(self.bert_dim + l1_size, output_size)

    def cos_similarity(premise: np.ndarray, label: np.ndarray):
        return np.dot(premise, label.T) / (np.linalg.norm(premise) * np.linalg.norm(label))
    
    def forward(self, premise):
        tokenized_premise = self.tokenizer.encode(premise, return_tensors='pt')
        bert_output = self.bert(tokenized_premise)["pooler_output"]

        sentence_bert_premise = self.similarity_model(premise)
        array = np.zeros((1, self.l1_size))
        for idx, label in enumerate(l1_labels):
            sentence_bert_l1 = self.similarity_model.encode(label)
            similarity = self.cos_similarity(sentence_bert_premise, sentence_bert_l1)
            array[0,idx] = similarity

        concat = torch.concat([bert_output,torch.tensor(array)], dim=1)
        output = self.linear(concat)

        return output

    

        




