"""
Tiny Recursion Model for Sudoku Solving
Implements a custom recursive neural network architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Tuple, List, Optional


class RecursiveCell(nn.Module):
    """A single recursive cell that processes Sudoku grids iteratively"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Recursive processing layers
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None, num_steps: int = 3):
        """
        Forward pass through the recursive cell
        
        Args:
            x: Input tensor of shape (batch_size, 9, 9, input_dim)
            hidden: Hidden state from previous iteration
            num_steps: Number of recursive steps to perform
        
        Returns:
            output: Processed tensor of shape (batch_size, 9, 9, output_dim)
            final_hidden: Final hidden state
        """
        batch_size, height, width, _ = x.shape
        
        # Project input to hidden dimension
        x_proj = self.input_proj(x.view(-1, x.size(-1)))  # (batch*81, hidden_dim)
        
        if hidden is None:
            hidden = torch.zeros(batch_size * height * width, self.hidden_dim, 
                               device=x.device, dtype=x.dtype)
        
        # Recursive processing
        for step in range(num_steps):
            hidden = self.cell(x_proj, hidden)
            hidden = self.layer_norm(hidden)
        
        # Project to output dimension
        output = self.output_proj(hidden)
        output = output.view(batch_size, height, width, -1)
        
        return output, hidden


class SudokuConstraintLayer(nn.Module):
    """Layer that enforces Sudoku constraints during processing"""
    
    def __init__(self, num_digits: int = 9):
        super().__init__()
        self.num_digits = num_digits
        
        # Constraint encoding networks
        self.row_encoder = nn.Linear(num_digits, num_digits)
        self.col_encoder = nn.Linear(num_digits, num_digits)
        self.box_encoder = nn.Linear(num_digits, num_digits)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Sudoku constraints
        
        Args:
            x: Input tensor of shape (batch_size, 9, 9, num_digits)
        
        Returns:
            Constraint-aware features
        """
        batch_size, height, width, channels = x.shape
        
        # Row constraints
        row_features = self.row_encoder(x)  # (batch, 9, 9, num_digits)
        
        # Column constraints
        col_features = self.col_encoder(x.transpose(1, 2).contiguous()).transpose(1, 2)
        
        # 3x3 box constraints
        box_features = self._apply_box_constraints(x)
        
        # Combine all constraint features
        constrained_features = x + row_features + col_features + box_features
        
        return constrained_features
    
    def _apply_box_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3x3 box constraints"""
        batch_size, height, width, channels = x.shape
        
        # Reshape to group 3x3 boxes
        # Each 3x3 box becomes a separate dimension
        boxes = x.reshape(batch_size, 3, 3, 3, 3, channels)  # (batch, 3, 3, 3, 3, channels)
        boxes = boxes.permute(0, 1, 3, 2, 4, 5)  # (batch, 3, 3, 3, 3, channels)
        boxes = boxes.contiguous().reshape(batch_size, 9, 9, channels)  # Flatten boxes
        
        # Apply box encoding
        box_encoded = self.box_encoder(boxes)
        
        # Reshape back to original format
        box_encoded = box_encoded.reshape(batch_size, 3, 3, 3, 3, channels)
        box_encoded = box_encoded.permute(0, 1, 3, 2, 4, 5)
        box_encoded = box_encoded.contiguous().reshape(batch_size, height, width, channels)
        
        return box_encoded


class TinyRecursionModel(pl.LightningModule):
    """Main model class implementing tiny recursive networks for Sudoku solving"""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_digits: int = 9,
        num_recursive_steps: int = 5,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_digits = num_digits
        self.num_recursive_steps = num_recursive_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Input embedding for digits (0-9)
        self.digit_embedding = nn.Embedding(10, hidden_dim)
        
        # Recursive processing layers
        self.recursive_layers = nn.ModuleList([
            RecursiveCell(
                input_dim=hidden_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim
            ) for i in range(num_layers)
        ])
        
        # Constraint layers
        self.constraint_layers = nn.ModuleList([
            SudokuConstraintLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_digits + 1)  # +1 for empty cell
        )
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_digits + 1)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_digits + 1)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_digits + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input Sudoku grid (batch_size, 9, 9) with values 0-9
        
        Returns:
            Predictions for each cell (batch_size, 9, 9, num_digits+1)
        """
        batch_size, height, width = x.shape
        
        # Embed input digits
        x_embedded = self.digit_embedding(x.long())  # (batch_size, 9, 9, hidden_dim)
        
        # Process through recursive layers
        hidden_states = []
        current_input = x_embedded
        
        for layer_idx, (recursive_layer, constraint_layer) in enumerate(
            zip(self.recursive_layers, self.constraint_layers)
        ):
            # Recursive processing
            processed, hidden = recursive_layer(
                current_input, 
                hidden_states[layer_idx-1] if layer_idx > 0 else None,
                self.num_recursive_steps
            )
            hidden_states.append(hidden)
            
            # Apply constraints
            constrained = constraint_layer(processed)
            current_input = constrained
        
        # Final prediction
        predictions = self.prediction_head(current_input)
        
        return predictions
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        puzzle, solution = batch
        
        # Forward pass
        predictions = self(puzzle)
        
        # Compute loss
        loss = F.cross_entropy(
            predictions.view(-1, self.num_digits + 1),
            solution.view(-1)
        )
        
        # Compute accuracy
        preds = torch.argmax(predictions, dim=-1)
        accuracy = self.train_accuracy(preds.view(-1), solution.view(-1))
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        puzzle, solution = batch
        
        # Forward pass
        predictions = self(puzzle)
        
        # Compute loss
        loss = F.cross_entropy(
            predictions.view(-1, self.num_digits + 1),
            solution.view(-1)
        )
        
        # Compute accuracy
        preds = torch.argmax(predictions, dim=-1)
        accuracy = self.val_accuracy(preds.view(-1), solution.view(-1))
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step"""
        puzzle, solution = batch
        
        # Forward pass
        predictions = self(puzzle)
        
        # Compute loss
        loss = F.cross_entropy(
            predictions.view(-1, self.num_digits + 1),
            solution.view(-1)
        )
        
        # Compute accuracy
        preds = torch.argmax(predictions, dim=-1)
        accuracy = self.test_accuracy(preds.view(-1), solution.view(-1))
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', accuracy, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Prediction step"""
        if isinstance(batch, tuple):
            puzzle = batch[0]
        else:
            puzzle = batch
        
        predictions = self(puzzle)
        return torch.argmax(predictions, dim=-1)