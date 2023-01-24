import sys

sys.path.append('../')
import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.utils import logging

from data.dataset import ValuesDataset, ValuesDataCollator
from evaluation import compute_metrics
from model_stringconcat import SimilarityModel
from utils import read_labels

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


if __name__ == "__main__":
    l2_labels, l1_labels, l1_to_l2_map = read_labels()
    traindata = ValuesDataset("training")
    evaldata = ValuesDataset("validation")
    collator = ValuesDataCollator()
    model = SimilarityModel(len(l2_labels), l1_labels, l1_to_l2_map)

    args = TrainingArguments(
        output_dir="results",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=100,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='macro-avg-f1score',
        disable_tqdm=False
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
