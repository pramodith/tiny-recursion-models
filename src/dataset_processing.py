from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

def get_sudoku_dataset(
    split: str = "train",
    num_samples: int = None,
    last_num_samples: int = None,
    bos_token_id: int = 10,
):
    """Load sudoku dataset.

    You can select either:
      - the first num_samples examples (via HF slice [:num_samples])
      - the last last_num_samples examples (manual slicing after load)
    If both provided, num_samples (first) takes precedence.
    """
    if num_samples is not None and last_num_samples is not None:
        # Prioritize first N semantics if both given.
        last_num_samples = None
    hf_split = split if num_samples is None else f"{split}[:{num_samples}]"
    dataset = load_dataset("sapientinc/sudoku-extreme-1k", split=hf_split)
    if last_num_samples is not None:
        # Slice last N examples
        total = len(dataset)
        start = max(0, total - last_num_samples)
        dataset = dataset.select(range(start, total))
    dataset = dataset.filter(lambda example: len(example["question"]) == 81)
    dataset = dataset.map(
        lambda example: {
            "question": example["question"].replace(".", "0"),
            "answer": example["answer"].replace(".", "0"),
        },
    )
    dataset = dataset.map(
        lambda example: {
            "question_input_ids": [bos_token_id] + [int(c) for c in example["question"]],
            "answer_input_ids": [bos_token_id] + [int(c) for c in example["answer"]],
        },
    )
    return dataset.select_columns(["question_input_ids", "answer_input_ids"])

def get_dataloader(
    split: str = "train",
    batch_size: int = 32,
    seed: int = 42,
    num_samples: int = None,
    last_num_samples: int = None,
    bos_token_id: int = 10,
):
    dataset = get_sudoku_dataset(
        split,
        num_samples=num_samples,
        last_num_samples=last_num_samples,
        bos_token_id=bos_token_id,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset.with_format("torch"), batch_size=batch_size, shuffle=True, generator=generator)

if __name__ == "__main__":
    dataloader = get_dataloader("train", batch_size=2)
    for batch in dataloader:
        print(batch)
        break