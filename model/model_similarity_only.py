import os.path

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityOnlyModel(nn.Module):
    def __init__(self, output_size, l1_labels, l1_to_l2_map, l1_exs):
        super(SimilarityOnlyModel, self).__init__()

        self.l1_exs_size = len(l1_exs)
        self.output_size = output_size

        if os.path.isfile("finetuned_sentence_transformer/config.json"):
            self.similarity_model = SentenceTransformer("finetuned_sentence_transformer/")
        else:
            print("Fine-tuned sentence-transformer not found. Defaulting to all-distilroberta-v1")
            self.similarity_model = SentenceTransformer('all-distilroberta-v1')
        self.hidden = nn.Linear(self.l1_exs_size, 768)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, self.output_size)

        for param in self.similarity_model.parameters():
            param.requires_grad = False

        self.l1_exs_embeds = self.similarity_model.encode(l1_exs, convert_to_tensor=True, normalize_embeddings=True)

    def forward(self, premises):
        sentence_bert_premises = self.similarity_model.encode(premises, convert_to_tensor=True, normalize_embeddings=True)
        similarity = torch.matmul(sentence_bert_premises, self.l1_exs_embeds.T)

        hidden = F.relu(self.hidden(similarity))
        output = self.linear(self.dropout(hidden))

        return output
