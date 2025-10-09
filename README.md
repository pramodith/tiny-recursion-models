# Tiny Recursion Models for Sudoku Solving

This repository implements a custom Deep Learning Architecture using PyTorch Lightning to train and test on the Sudoku dataset from HuggingFace. The project is inspired by "Less is More: Recursive Reasoning with Tiny Networks" and implements a recursive neural network architecture specifically designed for Sudoku puzzle solving.

## Features

- **Custom Recursive Architecture**: Implements a novel recursive neural network with constraint layers specifically designed for Sudoku solving
- **PyTorch Lightning Integration**: Full PyTorch Lightning implementation with proper data modules, training loops, and callbacks
- **HuggingFace Dataset Support**: Loads the `sapientinc/sudoku-extreme-1k` dataset with fallback to mock data for development
- **Comprehensive Evaluation**: Includes visualization and evaluation tools for model performance analysis

## Architecture

The `TinyRecursionModel` consists of:

1. **Recursive Cells**: Process Sudoku grids through multiple recursive steps using GRU cells
2. **Constraint Layers**: Enforce Sudoku rules (row, column, and 3x3 box constraints) during processing
3. **Multi-layer Processing**: Stack multiple recursive layers for deeper reasoning
4. **Prediction Head**: Final classification layer for digit prediction (0-9)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Test the setup**:
```bash
python test_setup.py
```

2. **Train the model**:
```bash
python train.py --max_epochs 50 --batch_size 32
```

3. **Evaluate a trained model**:
```bash
python evaluate.py --model_path ./checkpoints/tiny_recursion_sudoku/final_model.ckpt
```

### Configuration

You can modify training parameters using command line arguments or by editing `config.yaml`:

```bash
python train.py \
    --hidden_dim 64 \
    --num_recursive_steps 5 \
    --num_layers 3 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --max_epochs 50
```

### Dataset

The model is designed to work with the `sapientinc/sudoku-extreme-1k` dataset from HuggingFace. If the dataset is not accessible, the system automatically falls back to generated mock data for development and testing.

## Model Architecture Details

### Recursive Cell
- Uses GRU cells for temporal processing
- Applies layer normalization for training stability
- Configurable number of recursive steps

### Constraint Layer
- Enforces row constraints using linear transformations
- Applies column constraints through transposition
- Implements 3x3 box constraints with tensor reshaping

### Training Features
- Adam optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- TensorBoard logging for training visualization

## Project Structure

```
tiny-recursion-models/
├── src/
│   ├── data/
│   │   └── sudoku_datamodule.py    # Data loading and preprocessing
│   ├── models/
│   │   └── tiny_recursion_model.py # Main model architecture
│   └── utils/
│       ├── config.py               # Configuration utilities
│       └── sudoku_utils.py         # Sudoku-specific utilities
├── train.py                        # Training script
├── evaluate.py                     # Evaluation script
├── test_setup.py                   # Setup verification
├── config.yaml                     # Configuration file
└── requirements.txt                # Dependencies
```

## Results

The model achieves reasonable performance on Sudoku puzzle solving tasks. Training logs and model checkpoints are saved to the `./checkpoints` directory.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
