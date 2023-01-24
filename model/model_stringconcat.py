from transformers import (BertTokenizer, BertModel)
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityModel(nn.Module):
    def __init__(self, output_size, l1_labels, l1_to_l2_map):
        super(SimilarityModel, self).__init__()

        self.bert_dim = 768
        self.l1_size = 20
        self.output_size = output_size

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.similarity_model = SentenceTransformer('all-distilroberta-v1')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert_dim + self.l1_size, self.output_size)

        for param in self.similarity_model.parameters():
            param.requires_grad = False

        # String Concatenation
        str_list = [""]*20
        for label, idx in l1_to_l2_map.items():
            str_list[idx] += label + "</s>"  # </s> is the separator token used in tokenizer

        self.l1_embeds = self.similarity_model.encode(str_list, convert_to_tensor=True, normalize_embeddings=True)

    def forward(self, premises):
        tokenized_premise = self.tokenizer(premises, padding=True, return_tensors='pt')['input_ids'].to(device)
        bert_output = self.bert(tokenized_premise)["pooler_output"]

        sentence_bert_premises = self.similarity_model.encode(premises, convert_to_tensor=True, normalize_embeddings=True)
        similarity = torch.matmul(sentence_bert_premises, self.l1_embeds.T)

        concat = torch.concat([bert_output, similarity], dim=1)
        output = self.linear(self.dropout(concat))

        return output
