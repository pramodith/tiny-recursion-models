import random
import numpy as np
from transformers import ModernBertConfig
from typing import Optional
from torch import nn
import torch
from torch import optim
from model.lr_schedulers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from datetime import datetime
from math import ceil, sqrt
import os


# ----------------------------------------------------------------------
# Helper: truncated normal initialization similar to reference code.
# Wraps nn.init.trunc_normal_ for convenience.
# ----------------------------------------------------------------------
def trunc_normal_init_(tensor: torch.Tensor, std: float = 0.02, mean: float = 0.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=mean - 2*std, b=mean + 2*std)

# --- SwiGLU activation function ---
class SwiGLU(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        out = F.silu(gate) * up
        return self.down_proj(out)


class PatchedModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig, rms_norm_eps: float = 1e-5):
        super().__init__()
        # Use the original ModernBertModel for embeddings and other logic
        from transformers import ModernBertModel as OrigModernBertModel
        self.base = OrigModernBertModel(config)
        self.base.embeddings.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=None
        )
        # Initialize embedding weights with trunc_normal_init_
        trunc_normal_init_(
            self.base.embeddings.tok_embeddings.weight, std=1/sqrt(config.hidden_size), mean=0.0
        )
        self.base.embeddings.norm = nn.RMSNorm(config.hidden_size, eps=rms_norm_eps)        # Patch encoder layers to use SwiGLU
        for layer in self.base.layers:
           layer.attn_norm = nn.Identity()
           layer.mlp_norm = nn.RMSNorm(config.hidden_size, eps=rms_norm_eps)
           layer.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)
# ----------------------------------------------------------------------
# Set global random seed for reproducibility
# ----------------------------------------------------------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(42)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self):
        for k, v in self.model.state_dict().items():
            self.ema_state[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_ema_weights(self):
        self.model.load_state_dict(self.ema_state)


class CastedLinear(nn.Module):
    """Linear layer with dtype casting and truncated LeCun normal init.

    Weight initialized ~ N(0, 1/sqrt(in_features)) (truncated) as in some
    reasoning model heads; optional zero bias.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias=bias)

class TRMModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 11, 
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 81 + 1, # +1 for cls token
        seq_len: int = 81,
        n: int = 6,
        t: int = 3,
        num_supervisions: int = 16,
        ema_decay: float = 0.999,
        wandb_project: str = "tiny-recursion-models",
        wandb_run_name: str = None,
        do_log: bool = True,
    ):
        super().__init__()
        # TODO: Use swiglu activation in feedforward layers
        self.config = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * 1.5),
            num_hidden_layers=2,
            global_attn_every_n_layers=1,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            repad_logits_with_grad=True,
            pad_token_id=10,
            eos_token_id=10,
            bos_token_id=10,
            cls_token_id=10,
            sep_token_id=10,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        self.model = PatchedModernBertModel(config=self.config)
        # Use ModuleList so norms move with .to(device)
        self.post_layer_norm = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=1e-5) for _ in range(self.config.num_hidden_layers)
        ])

        # ------------------------------------------------------------------
        # State seeds (reference-style): instead of using torch.empty during
        # forward, we keep fixed (non-trainable) truncated normal seeds that
        # we broadcast to initialize y and z when they are first created.
        # This mimics reference code that stores H_init / L_init as buffers.
        # If later you want these to be learnable, convert to nn.Parameter.
        # ------------------------------------------------------------------
        with torch.no_grad():
            y_seed = torch.empty(self.seq_len + 1, hidden_size, device=self.device)
            z_seed = torch.empty(self.seq_len + 1, hidden_size, device=self.device)
            # Use normal_ with small std (like BERT style). Adjust if paper used std=1.
            y_seed.normal_(mean=0.0, std=0.02)
            z_seed.normal_(mean=0.0, std=0.02)
        self.register_buffer("y_seed", y_seed, persistent=True)
        self.register_buffer("z_seed", z_seed, persistent=True)

        # ------------------------------------------------------------------
        # LM Head using CastedLinear with truncated LeCun normal init.
        # Weight ~ N(0, 1/sqrt(in_features)) truncated to bounds (mean±2*std).
        # Bias omitted (bias=False) for parity with common LM heads.
        # ------------------------------------------------------------------
        self.lm_head = CastedLinear(hidden_size, vocab_size)
        self.q = CastedLinear(hidden_size, 1)
        self.num_latent_recursions = n
        self.num_solution_recursions = t
        self.num_supervisions = num_supervisions
        self.ce_loss = nn.CrossEntropyLoss()
        self.be_loss = nn.BCEWithLogitsLoss()
        self.active_train_step = 0

        # wandb setup
        if not do_log:
            wandb.init(mode="disabled")
        else:
            self.wandb_project = wandb_project
            self.wandb_run_name = wandb_run_name if wandb_run_name else datetime.now().isoformat()
            self.wandb_run = wandb.init(project=self.wandb_project, name=self.wandb_run_name, config={
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "max_position_embeddings": max_position_embeddings,
            "seq_len": seq_len,
            "num_latent_recursions": n,
            "num_solution_recursions": t,
            "num_supervisions": num_supervisions,
            "ema_decay": ema_decay,
        })
        self.to(self.device)
        self.ema = EMA(self, decay=ema_decay)

    
    def latent_recursion(self, x, y, z):
        position_ids = torch.arange(self.seq_len+1, device=self.device).unsqueeze(0).expand(x.size(0), -1)
        for _ in range(self.num_latent_recursions):
            z_input = x + y + z
            for layer_ind in range(len(self.model.base.layers)):
                z_input = self.model.base.layers[layer_ind](
                    z_input, 
                    position_ids=position_ids, 
                    attention_mask=None
                )[0]
                z_input = self.post_layer_norm[layer_ind](z_input)
                # with torch.no_grad():
                #     print(f"Mean of z_input after layer {layer_ind}: {z_input.mean().item()}")
            z = z_input
        y_input = y + z
        for layer_ind in range(len(self.model.base.layers)):
            y_input = self.model.base.layers[layer_ind](
                y_input, 
                position_ids=position_ids, 
                attention_mask=None
            )[0]
            y_input = self.post_layer_norm[layer_ind](y_input)
            # with torch.no_grad():
            #     print(f"Mean of y_input after layer {layer_ind}: {y_input.mean().item()}")
        return y_input, z_input
    
    def forward(self, batch, y=None, z=None):
        batch["question_input_ids"] = batch["question_input_ids"].to(self.device)
        batch["answer_input_ids"] = batch["answer_input_ids"].to(self.device)
        batch_size = len(batch["question_input_ids"])
        if not isinstance(z, torch.Tensor):
            # Broadcast seed to batch; clone to ensure no inadvertent in-place ops mutate buffer.
            z = self.z_seed.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if not isinstance(y, torch.Tensor):
            y = self.y_seed.unsqueeze(0).expand(batch_size, -1, -1).clone()
        x = self.model.base.embeddings(batch["question_input_ids"])
        with torch.no_grad():
            for _ in range(self.num_solution_recursions-1):
                y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)
        return y.detach(), z.detach(), self.lm_head(y[:, 1:]), self.q(y[:, 0]).squeeze(-1)

    def training_step(self, batch, y=None, z=None, batch_idx=None):
        y, z, logits, q = self(batch, y=y, z=z)
        stats = self._compute_batch_stats(batch, logits, q)
        loss = stats["loss"]
        # wandb logging (include tile/puzzle metrics)
        metrics = {
            "train/loss": stats["loss"].item(),
            "train/ce_loss": stats["ce_loss"].item(),
            "train/be_loss": stats["be_loss"].item(),
            "train/tile_accuracy": stats["tile_accuracy"],
            "train/puzzle_accuracy": stats["puzzle_accuracy"],
            "train/step": self.active_train_step,
            "train/q_mean": q.mean().item(),
        }
        print(f"Metrics at step {self.active_train_step}: {metrics}")
        wandb.log(metrics, step=self.active_train_step)
        return loss, y, z, q

    @torch.no_grad()
    def evaluate(self, dataloader, prefix: str = "val"):
        """Run validation over a dataloader.

        Computes:
        - validation loss (same formulation as training: CE + BE)
        - tile accuracy: percentage of fillable (masked in question) tiles predicted correctly
        - puzzle accuracy: percentage of puzzles with all fillable tiles predicted correctly
        """
        self.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_be_loss = 0.0
        total_tiles = 0
        total_correct_tiles = 0
        total_puzzles = 0
        solved_puzzles = 0
        for batch in dataloader:
            y, z, logits, q = self(batch)
            stats = self._compute_batch_stats(batch, logits, q)
            batch_size = batch["question_input_ids"].size(0)
            total_loss += stats["loss"].item() * batch_size
            total_ce_loss += stats["ce_loss"].item() * batch_size
            total_be_loss += stats["be_loss"].item() * batch_size
            total_correct_tiles += stats["correct_masked_tiles"]
            total_tiles += stats["num_masked_tiles"]
            solved_puzzles += stats["num_solved_puzzles"]
            total_puzzles += batch_size
        avg_loss = total_loss / total_puzzles
        avg_ce_loss = total_ce_loss / total_puzzles
        avg_be_loss = total_be_loss / total_puzzles
        tile_accuracy = total_correct_tiles / max(1, total_tiles)
        puzzle_accuracy = solved_puzzles / max(1, total_puzzles)
        metrics = {
            f"{prefix}/loss": avg_loss,
            f"{prefix}/ce_loss": avg_ce_loss,
            f"{prefix}/be_loss": avg_be_loss,
            f"{prefix}/tile_accuracy": tile_accuracy,
            f"{prefix}/puzzle_accuracy": puzzle_accuracy,
            "train/step": self.active_train_step,
            f"{prefix}/total_correct_tiles": total_correct_tiles,
            f"{prefix}/solved_puzzles": solved_puzzles,
        }
        print(f"{prefix.capitalize()} metrics at step {self.active_train_step}: {metrics}")
        wandb.log(metrics, step=self.active_train_step)
        self.train()
        return metrics

    def _compute_batch_stats(self, batch, logits, q):
        """Compute losses and accuracy metrics for a single batch.

        Returns dict with:
          loss, ce_loss, be_loss, tile_accuracy, puzzle_accuracy,
          correct_masked_tiles, num_masked_tiles, num_solved_puzzles
        """
        # Prepare CE labels with masking of given tiles
        ce_labels = batch["answer_input_ids"][:, 1:].reshape(-1).clone()
        given_mask_flat = batch["question_input_ids"][:, 1:].reshape(-1) != 0
        ce_labels[given_mask_flat] = -100
        ce_loss_val = self.ce_loss(logits.view(-1, self.config.vocab_size), ce_labels)
        y_preds = torch.argmax(logits, dim=-1)
        y_trues = batch["answer_input_ids"][:, 1:]
        # Binary head loss based on full sequence correctness (including already given tiles counted as trivially correct)
        be_loss_val = self.be_loss(q, torch.all(y_trues == y_preds, dim=-1).float())
        loss = ce_loss_val + be_loss_val
        # Tile accuracy - only positions that need filling (question tile == 0)
        fill_mask = batch["question_input_ids"][:, 1:] == 0
        correct_masked = (y_preds[fill_mask] == y_trues[fill_mask]).sum().item()
        num_masked = fill_mask.sum().item()
        tile_acc = correct_masked / num_masked if num_masked > 0 else 0.0
        # Puzzle solved if all fillable tiles are predicted correctly
        puzzle_solved_mask = torch.all((y_preds == y_trues) | (batch["question_input_ids"][:, 1:] != 0), dim=-1)
        num_solved = puzzle_solved_mask.sum().item()
        puzzle_acc = num_solved / batch["question_input_ids"].size(0)
        return {
            "loss": loss,
            "ce_loss": ce_loss_val,
            "be_loss": be_loss_val,
            "tile_accuracy": tile_acc,
            "puzzle_accuracy": puzzle_acc,
            "correct_masked_tiles": correct_masked,
            "num_masked_tiles": num_masked,
            "num_solved_puzzles": num_solved,
        }
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1.0)
        return optimizer

    def fit(
        self,
        dataloader,
        num_epochs: Optional[int] = 1,
        num_steps: Optional[int] = None,
        warmup_steps: int = 2000,
        validate_every: Optional[int] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        save_top_k: int = 2,
    ):
        do_end_training = False
        if num_epochs is None and num_steps is None:
            raise ValueError("Either num_epochs or num_steps must be provided.")
        num_epochs = ceil(num_steps / len(dataloader)) if num_steps is not None else num_epochs
        optimizer = self.configure_optimizers()
        total_steps = num_steps if num_steps is not None else num_epochs * len(dataloader)
        # Scheduler: 2k warmup steps, then cosine decay
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        # Prepare validation dataloader if requested
        # Checkpoint bookkeeping
        self._checkpoint_metric_key = "val/tile_accuracy"
        self._save_top_k = save_top_k
        self._saved_checkpoints = []  # list of tuples (metric, path)
        self._checkpoint_dir = os.path.join(checkpoint_dir, getattr(self, 'wandb_run_name', 'run'))
        if validate_every is not None:
            if val_dataloader is None:
                from dataset_processing import get_dataloader as get_data_loader_fn
                # First 2000 examples of test split per user request
                val_dataloader = get_data_loader_fn(split="test", batch_size=32, num_samples=2000)
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            print(f"Validation enabled: every {validate_every} steps over {len(val_dataloader.dataset)} samples; checkpoints at {self._checkpoint_dir}")
        for epoch in tqdm(range(num_epochs), desc="Epoch Number: "):
            wandb.log({"epoch": epoch})
            if do_end_training:
                break
            for batch in tqdm(dataloader, desc="Batch Number: "):
                if do_end_training:
                    break
                y, z = None, None
                for _ in range(self.num_supervisions):
                    loss, y, z, q = self.training_step(batch, y=y, z=z)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    # Log current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    wandb.log({"train/lr": current_lr}, step=self.active_train_step)
                    self.active_train_step += 1
                    optimizer.zero_grad()
                    # self.ema.update()
                    # Validation trigger
                    if validate_every is not None and (self.active_train_step % validate_every == 0):
                        val_metrics = self.evaluate(val_dataloader)
                        self._maybe_save_checkpoint(val_metrics)
                    if num_steps is not None and self.active_train_step >= num_steps:
                        do_end_training = True
                        break
                    # Sigmoid(0) = 0.5, so we check if q > 0 for all elements to stop 
                    if torch.all(q > 0):
                        break
        wandb.finish()
        return loss

    def _maybe_save_checkpoint(self, val_metrics: dict):
        """Save model if tile accuracy is among top-k.

        Maintains a list of saved checkpoints sorted descending by metric.
        Removes oldest (worst) if exceeding k.
        """
        metric = val_metrics.get(self._checkpoint_metric_key)
        if metric is None:
            return
        # Decide filename
        step = val_metrics.get("train/step", self.active_train_step)
        filename = f"step{step}_tileacc{metric:.4f}.pt"
        path = os.path.join(self._checkpoint_dir, filename)
        # Insert into list maintaining order
        self._saved_checkpoints.append((metric, path))
        self._saved_checkpoints.sort(key=lambda x: x[0], reverse=True)
        # If exceeds top-k, pop worst and delete file if existed
        while len(self._saved_checkpoints) > self._save_top_k:
            worst_metric, worst_path = self._saved_checkpoints.pop(-1)
            if os.path.exists(worst_path):
                try:
                    os.remove(worst_path)
                except OSError:
                    pass
        # If current path is within top-k (after sorting), save
        if any(p == path for _, p in self._saved_checkpoints[:self._save_top_k]):
            # Save state dict (could consider EMA weights optionally)
            torch.save({
                "step": step,
                "metric": metric,
                "model_state": self.state_dict(),
                "config": self.config.to_dict(),
            }, path)
            print(f"Saved checkpoint: {path} (tile_acc={metric:.4f})")

if __name__ == "__main__":
    model = TRMModel()
    print(model)