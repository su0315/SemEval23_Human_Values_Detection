import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging

sys.path.append('../')
from data.dataset import ValuesDataset, ValuesDataCollator
from model_l1_examples import SimilarityModel
from utils import read_labels

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    l2_labels, l1_labels, l1_to_l2_map, l1_exs = read_labels()
    testdata = ValuesDataset("test")
    collator = ValuesDataCollator()
    model = SimilarityModel(len(l2_labels), l1_labels, l1_to_l2_map, l1_exs).to(device)
    model.load_state_dict(torch.load("old_results/2 3e-5/pytorch_model.bin"))

    dataloader = DataLoader(testdata, batch_size=32, shuffle=False, collate_fn=collator.__call__)

    ids = []
    predictions = np.empty((0, len(l2_labels)), dtype=int)
    for batch in dataloader:
        ids.extend(batch['ids'])
        outputs = (model(batch['premises']).sigmoid() > 0.5).cpu().numpy().astype(int)
        predictions = np.concatenate((predictions, outputs), axis=0)

    print(len(ids), len(predictions))

    df = pd.DataFrame(columns=l2_labels, index=ids, data=predictions)
    df.index.name = 'Argument ID'
    print(df.shape)
    df.to_csv("predictions/test-predictions-2-3e-5.tsv", sep="\t")
