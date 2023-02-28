from enum import Enum

import torch
from sentence_transformers import InputExample
from torch.utils.data import Dataset
import pandas as pd
from transformers.data.data_collator import DataCollatorMixin
from main.utils import read_labels


class LabelsLevel(Enum):
    LEVEL1 = 1,
    LEVEL2 = 2,


class ValuesDataset(Dataset):

    def __init__(self, name: str, labels_level: LabelsLevel = LabelsLevel.LEVEL2):
        self.labels_level = labels_level
        if name != "full":
            self.arguments = pd.read_csv(f"data/arguments-{name}.tsv", delimiter="\t")
            if "test" not in name:
                self.l1_labels = pd.read_csv(f"data/level1-labels-{name}.tsv", delimiter="\t")
                self.l2_labels = pd.read_csv(f"data/labels-{name}.tsv", delimiter="\t")
            else:
                self.l1_labels, self.l2_labels = None, None
        else:
            self.arguments_train = pd.read_csv(f"data/arguments-training.tsv", delimiter="\t")
            self.arguments_val = pd.read_csv(f"data/arguments-validation.tsv", delimiter="\t")
            self.arguments = pd.concat((self.arguments_train, self.arguments_val), axis=0)
            self.l1_labels_train = pd.read_csv(f"data/level1-labels-training.tsv", delimiter="\t")
            self.l1_labels_val = pd.read_csv(f"data/level1-labels-validation.tsv", delimiter="\t")
            self.l1_labels = pd.concat((self.l1_labels_train, self.l1_labels_val), axis=0)
            self.l2_labels_train = pd.read_csv(f"data/labels-training.tsv", delimiter="\t")
            self.l2_labels_val = pd.read_csv(f"data/labels-validation.tsv", delimiter="\t")
            self.l2_labels = pd.concat((self.l2_labels_train, self.l2_labels_val), axis=0)

    def __len__(self):
        return len(self.arguments)

    def __getitem__(self, index):
        item = dict()
        item['id'] = self.arguments.iloc[index]['Argument ID']
        item['premises'] = self.arguments.iloc[index]['Premise']
        if self.labels_level == LabelsLevel.LEVEL2:
            item['label'] = torch.FloatTensor(self.l2_labels.iloc[index][1:]).view(1, -1) \
                if self.l2_labels is not None else torch.zeros((1, 20), dtype=torch.float32)
        else:
            item['label'] = torch.FloatTensor(self.l1_labels.iloc[index][1:]).view(1, -1) \
                if self.l1_labels is not None else torch.zeros((1, 54), dtype=torch.float32)
        return item


class ValuesDataCollator(DataCollatorMixin):
    def __call__(self, features, return_tensors='pt'):
        collated = dict()
        collated['ids'] = [input['id'] for input in features]
        collated['premises'] = [input['premises'] for input in features]
        collated['labels'] = [input['label'] for input in features]
        return collated


class SimilarityDataset(list):

    def __init__(self, name):
        super().__init__()
        arguments = pd.read_csv(f"data/arguments-{name}.tsv", delimiter="\t")
        _, _, _, concatted_examples = read_labels()
        l1_labels = pd.read_csv(f"data/level1-labels-{name}.tsv", delimiter="\t")

        merged = pd.merge(arguments, l1_labels, on="Argument ID").drop(["Argument ID", "Conclusion", "Stance"], axis=1)

        for row in merged.iloc:
            premise = row["Premise"]
            for label, value in zip(concatted_examples, row[1:]):
                if value == 1:
                    self.append(InputExample(texts=[premise, label], label=1.0))
                else:
                    self.append(InputExample(texts=[premise, label], label=-1.0))
