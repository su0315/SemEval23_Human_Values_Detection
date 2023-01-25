from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityModel(nn.Module):
    def __init__(self, output_size, l1_labels, l1_to_l2_map, l1_exs):
        super(SimilarityModel, self).__init__()

        self.l1_size = 20
        self.output_size = output_size

        self.similarity_model = SentenceTransformer('all-distilroberta-v1')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.l1_size, self.output_size)

        # String Concatenation
        self.concatted_l1_labels = [""]*20
        for label, idx in l1_to_l2_map.items():
            self.concatted_l1_labels[idx] += label + "</s>"  # </s> is the separator token used in tokenizer

    def forward(self, premises):
        l1_embeds = self.similarity_model.encode(self.concatted_l1_labels, convert_to_tensor=True, normalize_embeddings=True)
        sentence_bert_premises = self.similarity_model.encode(premises, convert_to_tensor=True, normalize_embeddings=True)
        similarity = torch.matmul(sentence_bert_premises, l1_embeds.T)
        
        output = self.linear(self.dropout(similarity))

        return output
