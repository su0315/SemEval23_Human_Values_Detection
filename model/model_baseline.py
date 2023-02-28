from transformers import (BertTokenizer, BertModel)
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineModel(nn.Module):
    def __init__(self, output_size, l1_labels, l1_to_l2_map, l1_exs):
        super(BaselineModel, self).__init__()
        self.bert_dim = 768
        self.output_size = output_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert_dim, self.output_size)

    def forward(self, premises):
        tokenized_premise = self.tokenizer(premises, padding=True, return_tensors='pt')['input_ids'].to(device)
        bert_output = self.bert(tokenized_premise)["pooler_output"]

        output = self.linear(self.dropout(bert_output))

        return output
