from pathlib import Path

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from config import MODEL_ID
from dataset import (
    load_audited_data,
    split_train_development,
    tokenize_example,
)
from modules import RadiologySummarizer


def make_loader(split, tokenizer, batch_size, shuffle):
    features = [
        tokenize_example(example, tokenizer)
        for example in split
    ] # Prepares tokenized examples 

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    ) # Reuses the tested collator for dynamic padding and -100 target padding

    return DataLoader(
        features,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    ) # Returns a dataloader with the features and collator


def run_epoch(model, loader, device, optimizer=None): # Processes every batch once
    # Passing an optimizer enables training, omitting it performs loss evaluation

    is_training = optimizer is not None
    model.train(is_training) # Selects training or eval behaviour

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        } # Move all tesnors to device

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            # Clear initial gradients if training

        with torch.set_grad_enabled(is_training):
            outputs = model(**batch) # Forward pass
            loss = outputs.loss 

            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss")
            # Check that loss is finite

            if is_training:
                loss.backward() # backpropagate
                torch.nn.utils.clip_grad_norm_( # Clipping, weight updates
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step() # Optimize

        token_count = (batch["labels"] != -100).sum().item() # Counts actual target tokens excluding padding
        total_loss += loss.item() * token_count 
        total_tokens += token_count # Accumulates token weighted loss

    return total_loss / total_tokens # Returns average loss per non-padding target token


def main():
    parser = argparse.ArgumentParser() 
    # Creates the parser


    parser.add_argument(
        "--mode",
        choices=["full", "lora"],
        default="full",
    ) # Defines a --mode option restricted to full or lora, defaulting to full

    parser.add_argument("--learning-rate", type=float, default=1e-4) # parses a numeric learning rate
    parser.add_argument("--run-name", default=None) # parses an optional checkpoint folder name
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--dev-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args() # Reads command line options, selected value is available as args.mode

    if min(args.epochs, args.train_size, args.dev_size, args.batch_size) <= 0:
        parser.error("Epochs, sample counts, and batch size must be positive")
        # Rejects invalid settings before loading the model

    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
        # Rejects invalid settings before loading the model


    torch.manual_seed(42) # Set seed for reproduction

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device, flush=True) # Select device

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = RadiologySummarizer(
        MODEL_ID,
        use_lora=args.mode == "lora",
    ).to(device)
    # Define model and tokenizer

    _, clean_train = load_audited_data() # Filter data 
    training, development = split_train_development(clean_train) # Keep the 5 percent for development

    training_subset = training.shuffle(seed=42).select(
        range(min(args.train_size, len(training)))
    )
    development_subset = development.shuffle(seed=42).select(
        range(min(args.dev_size, len(development)))
    )

    training_loader = make_loader(
        training_subset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
    ) # Create data loader for training data
    
    development_loader = make_loader(
        development_subset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
    ) # Create data loader for development

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
    )
    # Optimizer includes only trainable parameters. In LoRA mode, they are the adapters

    run_name = args.run_name or f"pilot_{args.mode}"

    checkpoint_dir = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / run_name
    ) # Save checkpoint

    checkpoint_dir.mkdir(parents=True, exist_ok=True) # Creates checkpoint directory
    resume_path = checkpoint_dir / "training_state.pt" 

    run_settings = { # Records the settings that must match when resuming. Omit the epoch line so we can extend a run
        "model_id": MODEL_ID,
        "mode": args.mode,
        "learning_rate": args.learning_rate,
        "train_size": len(training_subset),
        "dev_size": len(development_subset),
        "batch_size": args.batch_size,
        "seed": 42,
    }

    start_epoch = 1

    best_development_loss = float("inf")

    if args.resume:
        state = torch.load( # Reads local training state onto CPU
            resume_path,
            map_location="cpu",
            weights_only=True,
        )

        if state["settings"] != run_settings: # Prevents accidentally continuing a different experiment
            raise ValueError("Resume settings do not match the saved run")

        model.load_state_dict(state["model"]) # Restores the wrapper weights including adapters in LoRA mode
        optimizer.load_state_dict(state["optimizer"]) # Retains AdamW's accumulated update statistics

        start_epoch = state["completed_epoch"] + 1
        best_development_loss = state["best_development_loss"] # Preserves progress and checkpoint selection

        torch.set_rng_state(state["torch_rng"]) # Preserves random state

        if device.type == "cuda":
            if state["cuda_rng"] is None:
                raise ValueError("Checkpoint was not created on CUDA")
            torch.cuda.set_rng_state_all(state["cuda_rng"])

        if device.type == "mps":
            if state["mps_rng"] is None:
                raise ValueError("Checkpoint was not created on MPS")
            torch.mps.set_rng_state(state["mps_rng"])

        print(f"Resuming at epoch {start_epoch}", flush=True)

    elif resume_path.exists():
        raise FileExistsError(
            "Run already exists; use --resume or choose a new --run-name"
        ) # Continue an existing resumable run if available


    trainable_count = sum( # Confirms that the requested mode actually has the expected trainable count
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Learning rate:", args.learning_rate, flush=True)
    print("Training mode:", args.mode, flush=True)
    print("Trainable parameters:", f"{trainable_count:,}", flush=True)

    if not args.resume: # Measure initial loss only if its a fresh run
        initial_loss = run_epoch(model, development_loader, device)
        print(f"Initial development loss: {initial_loss:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        training_loss = run_epoch(
            model, training_loader, device, optimizer
        )
        development_loss = run_epoch(
            model, development_loader, device
        )

        print(
            f"Epoch {epoch}: training loss = {training_loss:.4f}, "
            f"development loss = {development_loss:.4f}",
            flush=True,
        )

        if development_loss < best_development_loss:
            best_development_loss = development_loss
            model.model.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(checkpoint_dir)
            print("Saved best pilot checkpoint", flush=True)

        # Finds best output and saves it

        recovery_state = {
            "settings": run_settings,
            "completed_epoch": epoch,
            "best_development_loss": best_development_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" else None
            ),
            "mps_rng": (
                torch.mps.get_rng_state()
                if device.type == "mps" else None
            ),
        }

        temporary_path = checkpoint_dir / "training_state.tmp"
        torch.save(recovery_state, temporary_path)
        temporary_path.replace(resume_path)

        print(f"Saved recovery state after epoch {epoch}", flush=True)


if __name__ == "__main__":
    main()