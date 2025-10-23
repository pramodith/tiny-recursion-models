"""CLI entrypoint for training TRMModel.

Example:
	python -m src.trainer \
		--train_batch_size 128 \
		--val_batch_size 128 \
		--num_steps 5000 \
		--validate_every 200 \
		--hidden_size 512 \
		--checkpoint_dir checkpoints \
		--save_top_k 2

If both --num_steps and --num_epochs are provided, num_steps takes precedence.
"""

import argparse
import torch
from model.trm import TRMModel
from dataset_processing import get_dataloader


def build_arg_parser():
	p = argparse.ArgumentParser(description="Train Tiny Recursion Model")
	# Data params
	p.add_argument("--train_split", default="train", help="HF dataset split for training")
	p.add_argument("--val_split", default="test_hard", help="HF dataset split for validation (first 2000 taken if num_val_samples set)")
	p.add_argument("--train_batch_size", type=int, default=128)
	p.add_argument("--val_batch_size", type=int, default=128)
	p.add_argument("--num_train_samples", type=int, default=None, help="Optional limit on number of training samples")
	p.add_argument("--num_val_samples", type=int, default=2000, help="Number of validation samples (first N) or None for full split")
	# Test params
	p.add_argument("--test_split", default="test_hard", help="Split name for test evaluation")
	p.add_argument("--test_batch_size", type=int, default=128)
	p.add_argument("--test_last_n", type=int, default=18000, help="Select last N examples from test split")
	p.add_argument("--run_test", action="store_true", help="Evaluate on test split after training")
	# Optimization params
	p.add_argument("--num_steps", type=int, default=2000, help="Total training steps (supervision loops counted)")
	p.add_argument("--num_epochs", type=int, default=None, help="Epochs (ignored if num_steps provided)")
	p.add_argument("--warmup_steps", type=int, default=2000)
	p.add_argument("--validate_every", type=int, default=200, help="Run validation every N steps; 0 disables validation")
	# Model architecture params
	p.add_argument("--vocab_size", type=int, default=11)
	p.add_argument("--hidden_size", type=int, default=512)
	p.add_argument("--num_attention_heads", type=int, default=8)
	p.add_argument("--seq_len", type=int, default=81)
	p.add_argument("--num_latent_recursions", type=int, default=6, dest="n")
	p.add_argument("--num_solution_recursions", type=int, default=3, dest="t")
	p.add_argument("--num_supervisions", type=int, default=16)
	# Checkpoint / logging
	p.add_argument("--checkpoint_dir", default="checkpoints")
	p.add_argument("--save_top_k", type=int, default=2)
	p.add_argument("--no_log", action="store_true", help="Disable wandb logging")
	p.add_argument("--wandb_project", default="tiny-recursion-models")
	p.add_argument("--wandb_run_name", default=None)
	# Device selection
	p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Execution device: auto picks cuda if available else cpu")
	return p


def main():
	parser = build_arg_parser()
	args = parser.parse_args()

	# Resolve device
	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	elif args.device == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but not available.")
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	if device.type == "cuda":
		# Ensure single GPU usage; optionally one could allow index override later
		torch.cuda.set_device(0)
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
	else:
		print("Using CPU")

	# Dataloaders
	train_loader = get_dataloader(
		args.train_split,
		batch_size=args.train_batch_size,
		num_samples=args.num_train_samples,
	)
	val_loader = None
	validate_every = args.validate_every if args.validate_every and args.validate_every > 0 else None
	if validate_every is not None:
		val_loader = get_dataloader(
			args.val_split,
			batch_size=args.val_batch_size,
			num_samples=args.num_val_samples,
		)

	model = TRMModel(
		vocab_size=args.vocab_size,
		hidden_size=args.hidden_size,
		num_attention_heads=args.num_attention_heads,
		seq_len=args.seq_len,
		n=args.n,
		t=args.t,
		num_supervisions=args.num_supervisions,
		wandb_project=args.wandb_project,
		wandb_run_name=args.wandb_run_name,
		do_log=not args.no_log,
	)
	# Move model to selected device (ensures proper placement even if internal logic already ran)
	model.to(device)

	fit_kwargs = dict(
		num_steps=args.num_steps if args.num_steps else None,
		num_epochs=args.num_epochs if (args.num_steps is None) else None,
		warmup_steps=args.warmup_steps,
		validate_every=validate_every,
		val_dataloader=val_loader,
		checkpoint_dir=args.checkpoint_dir,
		save_top_k=args.save_top_k,
	)
	model.fit(train_loader, **fit_kwargs)

	if args.run_test:
		print("\nRunning test evaluation...")
		test_loader = get_dataloader(
			args.test_split,
			batch_size=args.test_batch_size,
			last_num_samples=args.test_last_n,
		)
		model.evaluate(test_loader, prefix="test")


if __name__ == "__main__":
	main()