"""
Sudoku DataModule for PyTorch Lightning
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from datasets import load_dataset
import numpy as np
from typing import Optional, Tuple, List


class SudokuDataset(Dataset):
    """Custom Dataset for Sudoku puzzles"""
    
    def __init__(self, data: List[dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert puzzle and solution to tensors
        # Assuming the dataset has 'puzzle' and 'solution' fields
        if 'puzzle' in sample:
            puzzle = torch.tensor(sample['puzzle'], dtype=torch.float32)
            solution = torch.tensor(sample['solution'], dtype=torch.long)
        else:
            # Create mock data if structure is different
            puzzle = torch.randint(0, 10, (9, 9), dtype=torch.float32)
            solution = torch.randint(1, 10, (9, 9), dtype=torch.long)
        
        return puzzle, solution


class SudokuDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Sudoku dataset"""
    
    def __init__(
        self,
        dataset_name: str = "sapientinc/sudoku-extreme-1k",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Download data if needed. This is called only on 1 GPU/TPU in distributed"""
        try:
            # Try to load the actual dataset
            load_dataset(self.dataset_name)
        except Exception as e:
            print(f"Warning: Could not load dataset {self.dataset_name}: {e}")
            print("Will use mock data for development")
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages"""
        try:
            # Try to load the actual dataset
            dataset = load_dataset(self.dataset_name)
            
            # Check available splits
            if 'train' in dataset and 'test' in dataset:
                train_data = list(dataset['train'])
                test_data = list(dataset['test'])
            elif 'train' in dataset:
                # Split train into train/val
                train_data = list(dataset['train'])
                split_idx = int(len(train_data) * (1 - self.val_split))
                test_data = train_data[split_idx:]
                train_data = train_data[:split_idx]
            else:
                # Use the first available split
                first_split = list(dataset.keys())[0]
                all_data = list(dataset[first_split])
                split_idx = int(len(all_data) * 0.8)
                val_split_idx = int(len(all_data) * 0.9)
                train_data = all_data[:split_idx]
                test_data = all_data[val_split_idx:]
            
        except Exception as e:
            print(f"Using mock data due to: {e}")
            # Create mock data for development
            train_data = self._create_mock_data(800)
            test_data = self._create_mock_data(200)
        
        # Split train into train/val
        val_split_idx = int(len(train_data) * (1 - self.val_split))
        val_data = train_data[val_split_idx:]
        train_data = train_data[:val_split_idx]
        
        if stage == "fit" or stage is None:
            self.train_dataset = SudokuDataset(train_data)
            self.val_dataset = SudokuDataset(val_data)
        
        if stage == "test" or stage is None:
            self.test_dataset = SudokuDataset(test_data)
    
    def _create_mock_data(self, size: int) -> List[dict]:
        """Create mock sudoku data for development"""
        data = []
        for _ in range(size):
            # Create a simple mock puzzle (9x9 grid)
            puzzle = np.random.randint(0, 10, (9, 9)).tolist()  # 0 represents empty cells
            solution = np.random.randint(1, 10, (9, 9)).tolist()  # filled solution
            data.append({
                'puzzle': puzzle,
                'solution': solution
            })
        return data
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()