from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import INSTRUCTION
from dataset import load_audited_data
from modules import RadiologySummarizer


def main():
    checkpoint_dir = (
        Path(__file__).resolve().parent / "checkpoints" / "tiny_overfit"
    ) # Locates checkpoint relative to predict.py

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        ) # If directory doesn't exist, throw an error

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    ) # Select device

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = RadiologySummarizer(str(checkpoint_dir)).to(device)
    model.eval() # Define tokenizer, saved trained weights from the checkpoint, and activates eval mode

    _, clean_train = load_audited_data() # Loads filtered data

    for example_number, example in enumerate(
        
        clean_train.select(range(2)), 
        start=1,
    ):
        """
        This loop selects two diagnostic examples, tokenizes their instructed, reports,
        generated without gradients, and prints each result. 
        """
        inputs = tokenizer(
            INSTRUCTION + example["radiology_report"],
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        print(f"\nExample {example_number}")
        print("Report:", example["radiology_report"])
        print("Reference:", example["layman_report"])
        print("Generated:", summary)


if __name__ == "__main__":
    main()