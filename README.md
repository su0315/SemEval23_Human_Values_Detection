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
## Reference
Please cite the following paper:
Sumire Honda and Sebastian Wilharm. (2023). "[Noam Chomsky at SemEval-2023 task 4: Hierarchical similarity-aware model for human value detection](https://aclanthology.org/2023.semeval-1.188/)." In Proceedings of the 17th International Workshop on Semantic Eval-uation (SemEval-2023), Association for Computational Linguistics.
```bibtex
@inproceedings{honda-wilharm-2023-noam,
    title = "Noam {C}homsky at {S}em{E}val-2023 Task 4: Hierarchical Similarity-aware Model for Human Value Detection",
    author = "Honda, Sumire  and
      Wilharm, Sebastian",
    booktitle = "Proceedings of the The 17th International Workshop on Semantic Evaluation (SemEval-2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.semeval-1.188",
    pages = "1359--1364",
    abstract = "This paper presents a hierarchical similarity-aware approach for the SemEval-2023 task 4 human value detection behind arguments using SBERT. The approach takes similarity score as an additional source of information between the input arguments and the lower level of labels in a human value hierarchical dataset. Our similarity-aware model improved the similarity-agnostic baseline model, especially showing a significant increase in or the value categories with lowest scores by the baseline model.",
}
```
