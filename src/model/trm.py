import random
import numpy as np
from transformers import ModernBertModel, ModernBertConfig
from typing import Optional
from torch import nn
import torch
from torch import optim
from model.lr_schedulers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from datetime import datetime
from math import ceil

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

# ----------------------------------------------------------------------
# Helper: truncated normal initialization similar to reference code.
# Wraps nn.init.trunc_normal_ for convenience.
# ----------------------------------------------------------------------
def trunc_normal_init_(tensor: torch.Tensor, std: float = 0.02, mean: float = 0.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=mean - 2*std, b=mean + 2*std)


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
        vocab_size: int = 14, 
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
            pad_token_id=13,
            eos_token_id=11,
            bos_token_id=12,
            cls_token_id=12,
            sep_token_id=11,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

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
        self.model = ModernBertModel(config=self.config)

        # ------------------------------------------------------------------
        # LM Head using CastedLinear with truncated LeCun normal init.
        # Weight ~ N(0, 1/sqrt(in_features)) truncated to bounds (mean±2*std).
        # Bias omitted (bias=False) for parity with common LM heads.
        # ------------------------------------------------------------------
        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)
        self.q = nn.Linear(hidden_size, 1)
        self.num_latent_recursions = n
        self.num_solution_recursions = t
        self.num_supervisions = num_supervisions
        self.ce_loss = nn.CrossEntropyLoss()
        self.be_loss = nn.BCEWithLogitsLoss()
        self.ema = EMA(self, decay=ema_decay)
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
    
    def latent_recursion(self, x, y, z):
        position_ids = torch.arange(self.seq_len+1, device=self.device).unsqueeze(0).expand(x.size(0), -1)
        for _ in range(self.num_latent_recursions):
            z_input = x + y + z
            for layer_ind in range(len(self.model.layers)):
                z_input = self.model.layers[layer_ind](
                    z_input, 
                    position_ids=position_ids, 
                    attention_mask=None
                )[0]
            z = z_input
        y_input = y + z
        for layer_ind in range(len(self.model.layers)):
            y_input = self.model.layers[layer_ind](
                y_input, 
                position_ids=position_ids, 
                attention_mask=None
            )[0]
        return y_input, z_input
    
    def forward(self, batch, y=None, z=None):
        batch_size = len(batch["question_input_ids"])
        if not isinstance(z, torch.Tensor):
            # Broadcast seed to batch; clone to ensure no inadvertent in-place ops mutate buffer.
            z = self.z_seed.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if not isinstance(y, torch.Tensor):
            y = self.y_seed.unsqueeze(0).expand(batch_size, -1, -1).clone()
        x = self.model.embeddings(batch["question_input_ids"])
        with torch.no_grad():
            for _ in range(self.num_solution_recursions-1):
                y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)
        return y.detach(), z.detach(), self.lm_head(y[:, 1:]), self.q(y[:, 0]).squeeze(-1)

    def training_step(self, batch, y=None, z=None, batch_idx=None):
        y, z, logits, q = self(batch, y=y, z=z)
        # ------------------------------------------------------------------
        # Cross-entropy masking: ignore positions that were already given in
        # the original puzzle (question_input_ids != 0). We clone labels so
        # the dataloader's underlying tensor is not mutated (important when
        # using multiple supervisions / workers). CLS at index 0 is excluded
        # from logits so we slice from 1: for labels.
        # ------------------------------------------------------------------
        ce_labels = batch["answer_input_ids"][:, 1:].reshape(-1).clone()
        ce_labels[batch["question_input_ids"][:, 1:].reshape(-1) != 0] = -100
        ce_loss_val = self.ce_loss(logits.view(-1, self.config.vocab_size), ce_labels)
        y_preds = torch.argmax(logits, dim=-1)
        y_trues = batch["answer_input_ids"][:, 1:]
        # Sequence-level correctness (exact match across all predicted cells)
        # drives the binary Q-head target. Consider restricting to fillable
        # cells only or using a softer proportional accuracy target if this
        # proves too sparse early in training.
        be_loss_val = self.be_loss(q, torch.all(y_trues==y_preds, dim=-1).float())
        loss = ce_loss_val + be_loss_val

        # wandb logging
        metrics = {
            "train/loss": loss.item(),
            "train/ce_loss": ce_loss_val.item(),
            "train/be_loss": be_loss_val.item(),
            "train/step": self.active_train_step,
            "train/q_mean": q.mean().item(),
        }
        print(f"Metrics at step {self.active_train_step}: {metrics}")
        wandb.log(metrics, step=self.active_train_step)
        return loss, y, z, q
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1.0)
        return optimizer

    def fit(self, dataloader, num_epochs: Optional[int] = 1, num_steps: Optional[int] = None, warmup_steps: int = 2000):
        do_end_training = False
        if num_epochs is None and num_steps is None:
            raise ValueError("Either num_epochs or num_steps must be provided.")
        num_epochs = ceil(num_steps / len(dataloader)) if num_steps is not None else num_epochs
        optimizer = self.configure_optimizers()
        # Scheduler: 2k warmup steps, then cosine decay
        total_steps = num_steps if num_steps is not None else num_epochs * len(dataloader)
        warmup_steps = 2000
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
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
                    print(f"Current learning rate: {current_lr}")
                    wandb.log({"train/lr": current_lr}, step=self.active_train_step)
                    self.active_train_step += 1
                    optimizer.zero_grad()
                    if num_steps is not None and self.active_train_step >= num_steps:
                        do_end_training = True
                        break
                    # Sigmoid(0) = 0.5, so we check if q > 0 for all elements to stop 
                    if torch.all(q > 0):
                        break
        wandb.finish()
        return loss

if __name__ == "__main__":
    model = TRMModel()
    print(model)