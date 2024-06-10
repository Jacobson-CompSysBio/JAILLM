from from dataclasses import dataclass

@dataclass
class semantic_scholar:
    dataset: str = "allenai/peS2o"
    file: str = "examples/get_semantic_scholar.py"
    train_split: str = "train"
    test_split: str = "validation"