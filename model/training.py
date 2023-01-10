import pandas as pd
import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.utils import logging

from data.dataset import ValuesDataset, ValuesDataCollator
from evaluation import compute_metrics
from model import SimilarityModel

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        total_loss = 0
        outputs = []
        for premise, label in zip(inputs['premises'], inputs['labels']):
            label = label.to(device)
            output = model(premise=premise)
            outputs.append(output)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(output.view(-1, 20), label.view(-1, 20))
            total_loss += loss
        return (total_loss / len(inputs['premises']), torch.concat(outputs, dim=0)) if return_outputs \
            else total_loss / len(inputs['premises'])

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        labels = torch.concat(inputs['labels'], dim=0)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, outputs, labels)


l2_labels = pd.read_csv("data/labels-training.tsv", delimiter="\t").columns.to_numpy()[1:]
traindata = ValuesDataset("training")
evaldata = ValuesDataset("validation")
collator = ValuesDataCollator()
model = SimilarityModel(768, 54, 20)

args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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
trainer.evaluate()
