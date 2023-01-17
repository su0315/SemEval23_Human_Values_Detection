import json
from typing import List, Dict


def read_labels() -> (List[str], List[str], Dict[str, int]):
    """
    Reads in and returns labels of the different levels with their mapping.
    :return: list of l2 labels, list of l1 labels, mapping from l1 label to l2 index
    """
    l2_labels = []
    l1_labels = []
    l1_to_l2_map = {}

    with open("../data/value-categories.json", "r") as file:
        values = json.load(file)

    for l2_label in values:
        l2_index = len(l2_labels)
        l2_labels.append(l2_label)
        for l1_label in values[l2_label]:
            l1_labels.append(l1_label)
            l1_to_l2_map[l1_label] = l2_index

    return l2_labels, l1_labels, l1_to_l2_map


if __name__ == "__main__":
    l2_labels, l1_labels, l1_to_l2_map = read_labels()
    print(l2_labels, l1_labels, l1_to_l2_map)
