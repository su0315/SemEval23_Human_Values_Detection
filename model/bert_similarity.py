from transformers import (BertTokenizer, BertForSequenceClassification, Trainer)
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased")

# Read the Dataset
train_df = pd.read_csv("../data/arguments-training.tsv", delimiter="\t")
train_df = train_df['Premise']
np_train_df = train_df.to_numpy()

labels = pd.read_csv("../data/labels-training.tsv", delimiter="\t")
#print (labels)

# Tokenizing the Data
for string in np_train_df:
    encoded = tokenizer.encode(string, return_tensors='pt')
    print (encoded)
    break

