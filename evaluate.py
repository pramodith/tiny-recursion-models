"""
Evaluation script for Tiny Recursion Models
"""

import argparse
import torch
import pytorch_lightning as pl
from src.data.sudoku_datamodule import SudokuDataModule
from src.models.tiny_recursion_model import TinyRecursionModel
import numpy as np


def visualize_predictions(puzzle, solution, prediction):
    """Visualize a single Sudoku puzzle, solution, and prediction"""
    print("Puzzle:")
    print_sudoku(puzzle)
    print("\nGround Truth Solution:")
    print_sudoku(solution)
    print("\nModel Prediction:")
    print_sudoku(prediction)
    print("\nCorrect cells:", (solution == prediction).sum().item(), "/", solution.numel())
    print("-" * 50)


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


def evaluate_model(model_path: str, data_module: SudokuDataModule, num_samples: int = 5):
    """Evaluate the trained model and show sample predictions"""
    
    # Load the trained model
    model = TinyRecursionModel.load_from_checkpoint(model_path)
    model.eval()
    
    # Set up trainer for testing
    trainer = pl.Trainer(logger=False, enable_progress_bar=True)
    
    # Test the model
    print("Evaluating model on test set...")
    test_results = trainer.test(model, data_module)
    
    # Show sample predictions
    print(f"\nShowing {num_samples} sample predictions:")
    test_dataloader = data_module.test_dataloader()
    
    with torch.no_grad():
        for i, (puzzles, solutions) in enumerate(test_dataloader):
            if i >= num_samples:
                break
                
            predictions = model(puzzles)
            pred_digits = torch.argmax(predictions, dim=-1)
            
            # Show first puzzle from batch
            visualize_predictions(
                puzzles[0], 
                solutions[0], 
                pred_digits[0]
            )
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tiny Recursion Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Set up data module
    data_module = SudokuDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluate the model
    results = evaluate_model(args.model_path, data_module, args.num_samples)
    
    print("\nEvaluation Results:")
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()