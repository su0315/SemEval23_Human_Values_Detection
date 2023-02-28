# Hierarchical Similarity-aware Model for Human Value Detection
## noam-chomsky at SemEval23 Task 4
#### Authors: Sumire Honda, Sebastian Wilharm

---

This repository contains our submission to the SemEval-2023 Task 4: Human Value Detection.

Final macro F1-score on the official test is 0.46 which resulted in rank 19 out of 39.

---

To fine-tune the SBERT model run
``python main/sbert_finetune.py``

To train the final submission model, run
``python main/training.py``

To generate prediction file, run
``python main/predict.py -m MODEL_FILE``

---