from transformers import (BertTokenizer, BertModel, Trainer)
from transformers import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

logging.set_verbosity_error()

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
bert_model = BertModel.from_pretrained("bert-base-cased")
sentence_model = SentenceTransformer('all-distilroberta-v1')

# Read the Dataset
train_df = pd.read_csv("data/arguments-training.tsv", delimiter="\t")
train_df = train_df['Premise']
np_train_df = train_df.to_numpy()

labels = pd.read_csv("data/labels-training.tsv", delimiter="\t")

level1_df = pd.read_csv("data/level1-labels-training.tsv", delimiter="\t")
level1_labels = level1_df.columns[1:]


def similarity(premise: np.ndarray, label: np.ndarray):
    return np.dot(premise, label.T) / (np.linalg.norm(premise) * np.linalg.norm(label))


bert_labels = []
sentence_labels = []
for label in level1_labels:
    bert_labels.append(bert_model(**tokenizer(label, return_tensors='pt'))['pooler_output'].detach().numpy().flatten())
    sentence_labels.append(sentence_model.encode(label))

index = 0
premise = np_train_df[index]
print("Premise:")
print('\t', premise)
tokenized_premise = tokenizer.encode(premise, return_tensors='pt')
bert_premise = bert_model(tokenized_premise)['pooler_output'].detach().numpy().flatten()
sentence_premise = sentence_model.encode(premise)

print("Correct labels: ")
correct_indices = level1_df.iloc[index, 1:].to_numpy().nonzero()
print('\t', level1_labels[correct_indices].values)

similarity_list = []
for i in range(len(level1_labels)):
    similarity_list.append((level1_labels[i], similarity(bert_premise, bert_labels[i]), similarity(sentence_premise, sentence_labels[i])))

print('--------------------------')
print("Bert similarities:")
for s in sorted(similarity_list, key=lambda x: x[1], reverse=True):
    print(f"\t{s[0]:32} {s[1]:.4f}")
print('--------------------------')
print("Sentence transformer similarities:")
for s in sorted(similarity_list, key=lambda x: x[2], reverse=True):
    print(f"\t{s[0]:32} {s[2]:.4f}")
