from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

def get_sudoku_dataset(split:str="train", num_samples:int=None):
    split = split if num_samples is None else f"{split}[:{num_samples}]"
    dataset = load_dataset("sapientinc/sudoku-extreme-1k", split=split)
    dataset = dataset.filter(lambda example: len(example["question"])==81)
    # 0 will be used to indicate a masked token
    dataset = dataset.map(
        lambda example: {
            "question": example["question"].replace(".", "0"),
            "answer": example["answer"].replace(".", "0")
        },
    )
    # 10 is bos/cls token, will be used for determining if we should stop recursion early
    dataset = dataset.map(
        lambda example: {
            "question_input_ids": [10] + [int(c) for c in example["question"]],
            "answer_input_ids": [10] + [int(c) for c in example["answer"]]
        },
    )
    return dataset.select_columns(["question_input_ids", "answer_input_ids"])

def get_dataloader(split:str="train", batch_size:int=32, seed:int=42):
    dataset = get_sudoku_dataset(split)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset.with_format("torch"), batch_size=batch_size, shuffle=True, generator=generator)

if __name__ == "__main__":
    dataloader = get_dataloader("train", batch_size=2)
    for batch in dataloader:
        print(batch)
        break