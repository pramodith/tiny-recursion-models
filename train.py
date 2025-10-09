"""
Training script for Tiny Recursion Models on Sudoku dataset
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from src.data.sudoku_datamodule import SudokuDataModule
from src.models.tiny_recursion_model import TinyRecursionModel


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Tiny Recursion Model on Sudoku')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_recursive_steps', type=int, default=5, help='Number of recursive steps')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    
    # Training hyperparameters
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--accelerator', type=str, default='auto', help='Training accelerator')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices')
    
    # Logging and checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--experiment_name', type=str, default='tiny_recursion_sudoku', help='Experiment name')
    
    args = parser.parse_args()
    
    # Set up data module
    data_module = SudokuDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )
    
    # Set up model
    model = TinyRecursionModel(
        hidden_dim=args.hidden_dim,
        num_recursive_steps=args.num_recursive_steps,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name=args.experiment_name,
        default_hp_metric=False
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        filename='best-{epoch:02d}-{val_loss:.4f}',
        save_last=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.patience,
        verbose=True
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        enable_progress_bar=True,
        log_every_n_steps=50
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    print("Testing the model...")
    trainer.test(model, data_module)
    
    # Save final model
    final_model_path = f"{args.save_dir}/{args.experiment_name}/final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == '__main__':
    main()