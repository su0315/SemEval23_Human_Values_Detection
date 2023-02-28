import torch
from torch.utils.data import DataLoader
from transformers.utils import logging
from sentence_transformers import SentenceTransformer, losses, evaluation

from data.dataset import SimilarityDataset

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    traindata = SimilarityDataset("training")
    evaldata = SimilarityDataset("validation")

    similarity_model = SentenceTransformer('all-distilroberta-v1')

    train_dataloader = DataLoader(traindata, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(similarity_model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(evaldata, batch_size=512,
                                                                            show_progress_bar=True)

    similarity_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=20,
        output_path="finetuned_sentence_transformer"
    )

    print(evaluator(similarity_model, "finetuned_sentence_transformer"))
