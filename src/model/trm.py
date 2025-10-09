from transformers import ModernBertModel, ModernBertConfig
from typing import Optional
from torch import nn
import torch
from torch import optim
from tqdm import tqdm
import wandb
from datetime import datetime
from math import ceil

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
        self.seq_len = seq_len
        self.model = ModernBertModel(config=self.config)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.q = nn.Linear(hidden_size, 1)
        self.num_latent_recursions = n
        self.num_solution_recursions = t
        self.num_supervisions = num_supervisions
        self.ce_loss = nn.CrossEntropyLoss()
        self.be_loss = nn.BCEWithLogitsLoss()
        self.ema = EMA(self, decay=ema_decay)
        self.active_train_step = 0

        # wandb setup
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
        }, mode="offline")
    
    def latent_recursion(self, x, y, z):
        for _ in range(self.num_latent_recursions):
            z = self.model.layers(x + y + z)
        y = self.model.layers(y + z)
        return y, z
    
    def forward(self, batch):
        z = torch.empty((len(batch, self.seq_len, self.config.hidden_size)), device=self.device)
        y = torch.empty((len(batch, self.seq_len, self.config.hidden_size)), device=self.device)
        x = self.model.embeddings(batch["question_input_ids"])
        with torch.no_grad():
            for _ in range(self.num_solution_recursions-1):
                y, z = self.latent_recursion(x, y, z)
            y, z = self.latent_recursion(x, y, z)
        return y, z, self.model.lm_head(y[:, 1:]), self.q(y[:, 0]).squeeze(-1)
    
    def training_step(self, batch, batch_idx=None):
        y, z, logits, q = self(batch)
        # mask loss for numbers already present on the board
        batch["answer_input_ids"][batch["question_input_ids"]!=0] = -100
        ce_loss_val = self.ce_loss(logits.view(-1, self.config.vocab_size), batch["answer_input_ids"].view(-1))
        y_preds = torch.argmax(logits, dim=-1)
        y_trues = batch["answer_input_ids"][:, 1:]
        be_loss_val = self.be_loss(q, (y_trues==y_preds).float())
        loss = ce_loss_val + be_loss_val

        # wandb logging
        metrics = {
            "train/loss": loss.item(),
            "train/ce_loss": ce_loss_val.item(),
            "train/be_loss": be_loss_val.item(),
            "train/step": self.active_train_step,
            "train/q_mean": q.mean().item(),
        }
        wandb.log(metrics, step=self.active_train_step)
        return loss, q
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95))
        return optimizer

    def train(self, dataloader, num_epochs: Optional[int] = 1, num_steps: Optional[int] = None):
        do_end_training = False
        if num_epochs is None and num_steps is None:
            raise ValueError("Either num_epochs or num_steps must be provided.")
        num_epochs = ceil(num_steps / len(dataloader)) if num_steps is not None else num_epochs
        optimizer = self.configure_optimizers()
        for epoch in tqdm(range(num_epochs), desc="Epoch Number: "):
            wandb.log({"epoch": epoch})
            if do_end_training:
                break
            for batch in tqdm(dataloader, desc="Batch Number: "):
                if do_end_training:
                    break
                optimizer.zero_grad()
                for _ in range(self.num_supervisions):
                    loss, q = self.training_step(batch)
                    loss.backward()
                    optimizer.step()
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