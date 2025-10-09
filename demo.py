"""
Demo script for Tiny Recursion Model
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.tiny_recursion_model import TinyRecursionModel
from src.utils.sudoku_utils import generate_random_sudoku, print_sudoku
import numpy as np


def demo_model_inference():
    """Demonstrate model inference on a sample Sudoku puzzle"""
    print("🧩 Tiny Recursion Model for Sudoku Solving Demo\n")
    
    # Create a small model for demo
    model = TinyRecursionModel(
        hidden_dim=32,
        num_layers=2,
        num_recursive_steps=3
    )
    model.eval()
    
    # Generate a sample puzzle
    puzzle, solution = generate_random_sudoku()
    
    print("📝 Sample Sudoku Puzzle:")
    print_sudoku(puzzle)
    
    print("\n✅ Ground Truth Solution:")
    print_sudoku(solution)
    
    # Convert to tensor and predict
    puzzle_tensor = torch.tensor(puzzle, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        predictions = model(puzzle_tensor)
        predicted_digits = torch.argmax(predictions, dim=-1)[0]  # Remove batch dim
    
    print("\n🤖 Model Prediction (untrained):")
    print_sudoku(predicted_digits.numpy())
    
    # Calculate accuracy
    correct_cells = (predicted_digits.numpy() == solution).sum()
    total_cells = solution.size
    accuracy = correct_cells / total_cells
    
    print(f"\n📊 Accuracy: {correct_cells}/{total_cells} ({accuracy:.2%})")
    print("\n💡 Note: This is an untrained model. Train with 'python train.py' for better results!")
    

def demo_architecture_info():
    """Display information about the model architecture"""
    print("\n🏗️  Model Architecture Information:")
    
    model = TinyRecursionModel(hidden_dim=64, num_layers=3, num_recursive_steps=5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Hidden dimension: {model.hidden_dim}")
    print(f"Number of layers: {len(model.recursive_layers)}")
    print(f"Recursive steps: {model.num_recursive_steps}")
    
    # Show layer structure
    print("\n📋 Layer Structure:")
    for i, (name, module) in enumerate(model.named_children()):
        if hasattr(module, '__len__'):
            print(f"  {name}: {len(module)} layers")
        else:
            print(f"  {name}: {type(module).__name__}")


if __name__ == "__main__":
    demo_model_inference()
    demo_architecture_info()
    
    print("\n🚀 To train the model, run:")
    print("   python train.py --max_epochs 50")
    print("\n🔍 To evaluate a trained model, run:")
    print("   python evaluate.py --model_path ./checkpoints/tiny_recursion_sudoku/final_model.ckpt")