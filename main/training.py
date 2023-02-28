from argparse import ArgumentParser

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.utils import logging

from data.dataset import ValuesDataset, ValuesDataCollator, LabelsLevel
from evaluation import compute_metrics
from utils import read_labels

from model.model_baseline import BaselineModel
from model.model_similarity_only import SimilarityOnlyModel
from model.model_final import FinalModel

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(premises=inputs['premises'])
        loss_fct = nn.BCEWithLogitsLoss()
        labels = torch.concat(inputs['labels'], dim=0).to(device)
        loss = loss_fct(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss, None, None
        labels = torch.concat(inputs['labels'], dim=0)
        return loss, outputs, labels


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', default="final", choices=['baseline', 'similarity', 'final'])
    parser.add_argument('-d', '--data', default="full", choices=['train-val', 'full'])
    parser.add_argument('-l', '--labels', default="l2", choices=['l1', 'l2'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    l2_labels, l1_labels, l1_to_l2_map, l1_exs = read_labels()

    if args.labels == 'l1':
        labels = LabelsLevel.LEVEL1
    else:
        labels = LabelsLevel.LEVEL2

    if args.data == 'train-val':
        traindata = ValuesDataset("training", labels)
        evaldata = ValuesDataset("validation", labels)
    else:
        traindata = ValuesDataset("full", labels)
        evaldata = ValuesDataset("validation", labels)

    collator = ValuesDataCollator()

    if args.model == 'baseline':
        model = BaselineModel(len(l2_labels), l1_labels, l1_to_l2_map, l1_exs)
    elif args.model == 'similarity':
        model = SimilarityOnlyModel(len(l2_labels), l1_labels, l1_to_l2_map, l1_exs)
    else:
        model = FinalModel(len(l2_labels), l1_labels, l1_to_l2_map, l1_exs)

    args = TrainingArguments(
        output_dir="results",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=100,
        num_train_epochs=20,
        optim="adamw_torch",
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='macro-avg-f1score',
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    trainer = SimilarityTrainer(
        model,
        args,
        train_dataset=traindata,
        eval_dataset=evaldata,
        compute_metrics=lambda x: compute_metrics(x, l2_labels),
        data_collator=collator
    )

    trainer.train()
