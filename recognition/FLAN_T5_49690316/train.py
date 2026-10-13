from pathlib import Path

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
    torch.manual_seed(42) # Set seed for reproduction

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    ) # Select device

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = RadiologySummarizer(MODEL_ID).to(device) # Define model and tokenizer

    _, clean_train = load_audited_data() # Filter data 
    training, development = split_train_development(clean_train) # Keep the 5 percent for development

    pilot_training = training.shuffle(seed=42).select(range(128))
    pilot_development = development.shuffle(seed=42).select(range(32))

    training_loader = make_loader(
        pilot_training, tokenizer, batch_size=2, shuffle=True
    ) # Create data loader for training data
    development_loader = make_loader(
        pilot_development, tokenizer, batch_size=2, shuffle=False
    ) # Create data loader for development

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Optimizer

    checkpoint_dir = (
        Path(__file__).resolve().parent / "checkpoints" / "pilot_full"
    ) # Save checkpoint

    best_development_loss = float("inf")

    initial_loss = run_epoch(model, development_loader, device)
    print(f"Initial development loss: {initial_loss:.4f}", flush=True)

    for epoch in range(1, 3):
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


if __name__ == "__main__":
    main()