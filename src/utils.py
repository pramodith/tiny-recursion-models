import sys
from typing import Optional

# ANSI color codes
_GREEN = "\033[92m"
_RED = "\033[91m"
_DIM = "\033[2m"
_RESET = "\033[0m"

def _validate_board(name: str, board: str):
    if not isinstance(board, str):
        raise TypeError(f"{name} must be a string; got {type(board)}")
    if len(board) != 81:
        raise ValueError(f"{name} must be length 81; got {len(board)}")
    if any(c not in "0123456789" for c in board):
        raise ValueError(f"{name} must contain only digits 0-9")


def visualize_sudoku(
    question: str,
    answer: str,
    proposed_answer: Optional[str] = None,
    stream = None,
    show_coords: bool = False,
    zero_char: str = ".",
    color: bool = True,
):
    """Visualize a Sudoku board.

    Parameters
    ----------
    question : str
        Puzzle string of length 81 (0 denotes blank originally). Used to mark given clues.
    answer : str
        Ground truth solution of length 81.
    proposed_answer : Optional[str]
        Model/user proposed solution of length 81. If provided, cells are colored:
          - Green if proposed digit matches answer.
          - Red if it differs.
        Given clues (non-zero in question) are dimmed.
    stream : file-like
        Where to write output (defaults to sys.stdout).
    show_coords : bool
        If True, prints row/col headers for easier debugging.
    zero_char : str
        Character to display for zeros when showing the question puzzle.
    color : bool
        Enable ANSI colors.
    """
    stream = stream or sys.stdout

    _validate_board("question", question)
    _validate_board("answer", answer)
    if proposed_answer is not None:
        _validate_board("proposed_answer", proposed_answer)
    else:
        proposed_answer = answer  # For unified iteration; no color diff.

    def style(cell_idx: int, q_digit: str, a_digit: str, p_digit: str) -> str:
        # Base display char: if showing puzzle content, use q_digit or zero placeholder
        shown_digit = p_digit if proposed_answer is not None else a_digit
        # For blanks in question, display either proposed/answer digit; for given digits, show that digit.
        if q_digit != "0":
            shown_digit = q_digit  # Preserve given clue appearance
        elif shown_digit == "0":
            shown_digit = zero_char

        if not color:
            return shown_digit

        # If the cell was a given clue, dim it.
        if q_digit != "0":
            return f"{_DIM}{shown_digit}{_RESET}"

        # If we're evaluating a proposed answer.
        if proposed_answer is not None:
            if p_digit == a_digit and p_digit != "0":
                return f"{_GREEN}{shown_digit}{_RESET}"
            if p_digit != a_digit and p_digit != "0":
                return f"{_RED}{shown_digit}{_RESET}"
        return shown_digit

    # Build lines
    lines = []
    if show_coords:
        header = "    " + " ".join(str(c) for c in range(1, 10))
        lines.append(header)

    for r in range(9):
        row_cells = []
        for c in range(9):
            idx = r * 9 + c
            qd = question[idx]
            ad = answer[idx]
            pd = proposed_answer[idx]
            row_cells.append(style(idx, qd, ad, pd))
        row_str = " ".join(row_cells[0:3]) + " | " + " ".join(row_cells[3:6]) + " | " + " ".join(row_cells[6:9])
        if show_coords:
            lines.append(f"R{r+1}: {row_str}")
        else:
            lines.append(row_str)
        if r in (2,5):
            lines.append("------+-------+------")

    stream.write("\n".join(lines) + "\n")
    stream.flush()

__all__ = ["visualize_sudoku"]
