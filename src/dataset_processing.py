from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from utils import visualize_sudoku
import numpy as np

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

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def sudoku_collate_fn(
    samples,
    apply_shuffle: bool = True,
    shuffle_prob: float = 1.0,
):
    """Custom collate function for Sudoku batches.

    Steps per sample:
      1. Take question/answer input ids (shape = (82,)) where index 0 is BOS.
      2. Remove BOS, reshape remaining 81 digits -> (9, 9) boards.
      3. Optionally apply `shuffle_sudoku` (position + digit remap) with probability `shuffle_prob`.
      4. Re-flatten boards row-major and prepend BOS again.

    Returns a dict of stacked tensors (batch, 82).

    Notes:
      - BOS token is preserved and *not* transformed.
      - Assumes digits are 0-9 and length after removing BOS is exactly 81.
    """
    questions = []
    answers = []

    # Use numpy random for augmentation probability check; could be seeded externally if needed.
    for sample in samples:
        q_ids = sample["question_input_ids"]
        a_ids = sample["answer_input_ids"]

        # Ensure torch Tensor for consistent dtype handling.
        if not torch.is_tensor(q_ids):
            q_ids = torch.tensor(q_ids)
        if not torch.is_tensor(a_ids):
            a_ids = torch.tensor(a_ids)

        bos_q = q_ids[0].item()
        bos_a = a_ids[0].item()
        q_flat = q_ids[1:]
        a_flat = a_ids[1:]

        # Sanity checks.
        assert q_flat.numel() == 81, f"Expected 81 question digits, got {q_flat.numel()}"
        assert a_flat.numel() == 81, f"Expected 81 answer digits, got {a_flat.numel()}"

        board = q_flat.reshape(9, 9).numpy()
        solution = a_flat.reshape(9, 9).numpy()

        if apply_shuffle and (shuffle_prob >= 1.0 or np.random.rand() < shuffle_prob):
            board, solution = shuffle_sudoku(board, solution)

        # Reassemble tensors with BOS at front.
        q_new = torch.tensor([bos_q] + board.flatten().tolist(), dtype=q_ids.dtype)
        a_new = torch.tensor([bos_a] + solution.flatten().tolist(), dtype=a_ids.dtype)

        questions.append(q_new)
        answers.append(a_new)

    return {
        "question_input_ids": torch.stack(questions, dim=0),
        "answer_input_ids": torch.stack(answers, dim=0),
    }


def get_dataloader(
    split: str = "train",
    batch_size: int = 32,
    seed: int = 42,
    num_samples: int = None,
    last_num_samples: int = None,
    bos_token_id: int = 10,
    apply_shuffle: bool = False,
    shuffle_prob: float = 1.0,
):
    dataset = get_sudoku_dataset(
        split,
        num_samples=num_samples,
        last_num_samples=last_num_samples,
        bos_token_id=bos_token_id,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    collate_fn = None
    if apply_shuffle:
        # Wrap to bind parameters without partial import.
        def _collate(samples):
            return sudoku_collate_fn(samples, apply_shuffle=True, shuffle_prob=shuffle_prob)
        collate_fn = _collate
    return DataLoader(
        dataset.with_format("torch"),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )

if __name__ == "__main__":
    dataloader = get_dataloader("train", batch_size=2, apply_shuffle=True, shuffle_prob=1.0)
    for batch in dataloader:
        print("Batch question shape:", batch["question_input_ids"].shape)
        print("Batch answer shape:", batch["answer_input_ids"].shape)
        # Take first sample and convert back to 81-char strings (drop BOS at index 0)
        q_tensor = batch["question_input_ids"][0][1:]  # (81,)
        a_tensor = batch["answer_input_ids"][0][1:]  # (81,)
        question_str = "".join(str(int(x.item())) for x in q_tensor)
        answer_str = "".join(str(int(x.item())) for x in a_tensor)
        print("\nVisualizing first (possibly shuffled) board:")
        visualize_sudoku(question_str, answer_str, proposed_answer=None, show_coords=False)
        break