"""
Utility functions for Sudoku processing
"""

import torch
import numpy as np
from typing import List, Tuple


def is_valid_sudoku(grid: np.ndarray) -> bool:
    """
    Check if a 9x9 Sudoku grid is valid
    
    Args:
        grid: 9x9 numpy array representing the Sudoku grid
        
    Returns:
        True if valid, False otherwise
    """
    # Check rows
    for row in grid:
        if not is_valid_unit(row):
            return False
    
    # Check columns
    for col in range(9):
        if not is_valid_unit(grid[:, col]):
            return False
    
    # Check 3x3 boxes
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = grid[box_row:box_row+3, box_col:box_col+3].flatten()
            if not is_valid_unit(box):
                return False
    
    return True


def is_valid_unit(unit: np.ndarray) -> bool:
    """
    Check if a unit (row, column, or box) contains no duplicates
    
    Args:
        unit: 1D array representing a Sudoku unit
        
    Returns:
        True if valid, False otherwise
    """
    # Filter out zeros (empty cells)
    filled_cells = unit[unit != 0]
    
    # Check for duplicates
    return len(filled_cells) == len(np.unique(filled_cells))


def generate_random_sudoku() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random Sudoku puzzle and its solution
    Note: This is a simplified generator for development purposes
    
    Returns:
        puzzle: 9x9 array with some cells filled (0 represents empty)
        solution: 9x9 array with complete solution
    """
    # Create a complete valid Sudoku (simplified approach)
    base = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [9, 1, 2, 3, 4, 5, 6, 7, 8]
    ])
    
    # Shuffle rows within each 3x3 band and columns within each 3x3 stack
    solution = base.copy()
    
    # Create puzzle by removing some numbers
    puzzle = solution.copy()
    num_to_remove = np.random.randint(40, 60)  # Remove 40-60 cells
    positions = np.random.choice(81, num_to_remove, replace=False)
    
    for pos in positions:
        row, col = pos // 9, pos % 9
        puzzle[row, col] = 0
    
    return puzzle, solution


def calculate_puzzle_difficulty(puzzle: np.ndarray) -> str:
    """
    Estimate puzzle difficulty based on number of filled cells
    
    Args:
        puzzle: 9x9 Sudoku puzzle array
        
    Returns:
        Difficulty level as string
    """
    filled_cells = np.count_nonzero(puzzle)
    
    if filled_cells >= 35:
        return "Easy"
    elif filled_cells >= 30:
        return "Medium"
    elif filled_cells >= 25:
        return "Hard"
    else:
        return "Expert"


def puzzle_to_tensor(puzzle: np.ndarray) -> torch.Tensor:
    """Convert numpy puzzle to PyTorch tensor"""
    return torch.tensor(puzzle, dtype=torch.float32)


def tensor_to_puzzle(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy puzzle"""
    return tensor.cpu().numpy().astype(int)


def print_sudoku(grid):
    """Pretty print a 9x9 Sudoku grid"""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("| ", end="")
            if j == 8:
                print(grid[i][j])
            else:
                print(str(grid[i][j]) + " ", end="")