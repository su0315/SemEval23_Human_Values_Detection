import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers.data.data_collator import DataCollatorMixin


class ValuesDataset(Dataset):

    def __init__(self, name):
        self.arguments = pd.read_csv(f"../data/arguments-{name}.tsv", delimiter="\t")
        if name != "test":
            self.l1_labels = pd.read_csv(f"../data/level1-labels-{name}.tsv", delimiter="\t")
            self.l2_labels = pd.read_csv(f"../data/labels-{name}.tsv", delimiter="\t")

    def __len__(self):
        return len(self.arguments)

    def __getitem__(self, index):
        item = dict()
        item['premises'] = self.arguments.iloc[index]['Premise']
        item['label'] = torch.FloatTensor(self.l2_labels.iloc[index][1:]).view(1, -1)
        return item


class ValuesDataCollator(DataCollatorMixin):
    def __call__(self, features, return_tensors='pt'):
        collated = dict()
        collated['premises'] = [input['premises'] for input in features]
        collated['labels'] = [input['label'] for input in features]
        return collated
