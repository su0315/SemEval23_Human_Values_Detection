from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging

from data.dataset import ValuesDataset, ValuesDataCollator, LabelsLevel
from model.model_final import FinalModel
from utils import read_labels

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_parse():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help="Path to trained model")
    parser.add_argument('-d', '--data', default="test",
                        choices=["training", "validation", "validation-zhihu", "test", "test-nahjalbalagha"])
    parser.add_argument('-l', '--labels', default="l2", choices=['l1', 'l2'])
    return parser.parse_args()


if __name__ == "__main__":
    l2_labels, l1_labels, l1_to_l2_map, l1_exs = read_labels()
    args = args_parse()

    if args.labels == 'l1':
        labels = LabelsLevel.LEVEL1
    else:
        labels = LabelsLevel.LEVEL2

    testdata = ValuesDataset(args.data, labels)
    collator = ValuesDataCollator()

    model = FinalModel(len(l2_labels), l1_labels, l1_to_l2_map, l1_exs)
    model.load_state_dict(torch.load(args.model))

    dataloader = DataLoader(testdata, batch_size=32, shuffle=False, collate_fn=collator.__call__)

    ids = []
    predictions = np.empty((0, len(l2_labels)), dtype=int)
    for batch in tqdm(dataloader):
        ids.extend(batch['ids'])
        outputs = (model(batch['premises']).sigmoid() > 0.5).cpu().numpy().astype(int)
        predictions = np.concatenate((predictions, outputs), axis=0)

    print(len(ids), len(predictions))

    df = pd.DataFrame(columns=l2_labels, index=ids, data=predictions)
    df.index.name = 'Argument ID'
    print(df.shape)
    df.to_csv("predictions/predictions.tsv", sep="\t")
