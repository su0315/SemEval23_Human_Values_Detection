from torch.utils.data import Dataset
import pandas as pd


class ValuesDataset(Dataset):

    def __init__(self, name):
        self.arguments = pd.read_csv(f"data/arguments-{name}.tsv", delimiter="\t")
        if name != "test":
            self.l1_labels = pd.read_csv(f"data/level1-labels-{name}.tsv", delimiter="\t")
            self.l2_labels = pd.read_csv(f"data/labels-{name}.tsv", delimiter="\t")

    def __len__(self):
        return len(self.arguments)

    def __getitem__(self, index):
        return self.arguments.iloc[index], self.l1_labels.iloc[index], self.l2_labels.iloc[index]
