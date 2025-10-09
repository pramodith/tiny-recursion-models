"""
Test script to verify the setup works correctly
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.sudoku_datamodule import SudokuDataModule
from src.models.tiny_recursion_model import TinyRecursionModel
from src.utils.sudoku_utils import generate_random_sudoku, is_valid_sudoku


def test_data_module():
    """Test the data module"""
    print("Testing SudokuDataModule...")
    
    dm = SudokuDataModule(batch_size=4, num_workers=0)  # Use 0 workers for testing
    dm.setup()
    
    # Test train dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    puzzles, solutions = batch
    
    print(f"Batch shape: puzzles={puzzles.shape}, solutions={solutions.shape}")
    print(f"Puzzle range: [{puzzles.min():.1f}, {puzzles.max():.1f}]")
    print(f"Solution range: [{solutions.min():.1f}, {solutions.max():.1f}]")
    
    return True


def test_model():
    """Test the model"""
    print("Testing TinyRecursionModel...")
    
    model = TinyRecursionModel(
        hidden_dim=32,  # Smaller for testing
        num_layers=2,
        num_recursive_steps=3
    )
    
    # Create dummy batch
    batch_size = 4
    puzzles = torch.randint(0, 10, (batch_size, 9, 9), dtype=torch.float32)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(puzzles)
    
    print(f"Input shape: {puzzles.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Expected output shape: ({batch_size}, 9, 9, 10)")
    
    assert predictions.shape == (batch_size, 9, 9, 10), f"Wrong output shape: {predictions.shape}"
    
    return True


def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    puzzle, solution = generate_random_sudoku()
    print(f"Generated puzzle shape: {puzzle.shape}")
    print(f"Generated solution shape: {solution.shape}")
    
    print(f"Puzzle is valid: {is_valid_sudoku(puzzle)}")
    print(f"Solution is valid: {is_valid_sudoku(solution)}")
    
    print("Sample puzzle:")
    print(puzzle[:3, :3])  # Show 3x3 corner
    
    return True


def main():
    """Run all tests"""
    print("Running setup tests...\n")
    
    try:
        test_utils()
        print("✓ Utils test passed\n")
        
        test_data_module()
        print("✓ Data module test passed\n")
        
        test_model()
        print("✓ Model test passed\n")
        
        print("🎉 All tests passed! Setup is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)