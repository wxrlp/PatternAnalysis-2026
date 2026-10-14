from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import MODEL_ID, INSTRUCTION
from dataset import load_audited_data, split_train_development
from modules import RadiologySummarizer

import nltk
import numpy as np
from rouge_score import rouge_scorer

import argparse
import json


def generate_summaries(model_path, reports, device, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RadiologySummarizer(str(model_path)).to(device) 
    model.eval() # Define tokenizer, model, and set to evaluation mode

    summaries = []

    for index, report in enumerate(reports, start=1): # Processes one report at a time
        inputs = tokenizer(
            INSTRUCTION + report,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        summaries.append(
            tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ) # Decodes each output and adds it to the result
        )

        if index % 25 == 0 or index == len(reports):
            print(
                f"{model_path}: generated {index}/{len(reports)} summaries",
                flush=True,
            )

    return summaries # Returns the list of summary results



def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    ) # Configure four metrics
#     # ROUGE-1	Individual words
      # ROUGE-2	Consecutive two-word phrases
      # ROUGE-L	The longest matching sequence of words in the same order; they don’t need to be adjacent
      # ROUGE-Lsum	A summary-level version of ROUGE-L that accounts for sentence boundaries

    scores = {
        name: []
        for name in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    } # Creates empty list for each metric

    for prediction, reference in zip(predictions, references):
        prediction_sentences = "\n".join(nltk.sent_tokenize(prediction))
        # Splits prediction into sentences, then joins them with newline characters.

        reference_sentences = "\n".join(nltk.sent_tokenize(reference))
        # Splits references into sentences, then joins them with newline characters.

        example_scores = scorer.score(
            reference_sentences,
            prediction_sentences,
        ) # Calculate scores with reference first and prediction second

        for name, score in example_scores.items():
            scores[name].append(score.fmeasure)
            # Stores each metric's F1 score, which combines overlap precision and recall

    return { # Returns arithmetic mean of each metric across samples, on a 0 to 1 scale
        name: float(np.mean(values))
        for name, values in scores.items()
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    if args.samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Sample count and output token limit must be positive")

    project_dir = Path(__file__).resolve().parent
    checkpoint_dir = project_dir / "checkpoints" / args.checkpoint
    output_dir = project_dir / "results" / args.output_name

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir}")

    if output_dir.exists():
        raise FileExistsError(f"Choose a new output name: {output_dir}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Evaluation device:", device, flush=True)

    _, clean_train = load_audited_data()
    _, development = split_train_development(clean_train) # Filter data and split into training/development

    if args.samples > len(development):
        parser.error("Requested more examples than the development split contains")

    # Select a reproducible development subset of the requested size
    examples = development.shuffle(seed=42).select(range(args.samples))
    reports = examples["radiology_report"]
    references = examples["layman_report"]

    baseline_summaries = generate_summaries(
        MODEL_ID, reports, device, args.max_new_tokens
    )
    trained_summaries = generate_summaries(
        checkpoint_dir, reports, device, args.max_new_tokens
    )
    # Generates from the original and trained checkpoints using the same reports

    metrics = {
        "baseline": calculate_rouge(baseline_summaries, references),
        "trained": calculate_rouge(trained_summaries, references),
        "by_source": {},
    }

    # Report each source separately so aggregate scores do not hide differences
    for source in sorted(set(examples["source"])):
        indices = [
            index
            for index, value in enumerate(examples["source"])
            if value == source
        ]

        source_references = [references[index] for index in indices]

        metrics["by_source"][source] = {
            "count": len(indices),
            "baseline": calculate_rouge(
                [baseline_summaries[index] for index in indices],
                source_references,
            ),
            "trained": calculate_rouge(
                [trained_summaries[index] for index in indices],
                source_references,
            ),
        }

    output_dir.mkdir(parents=True)

    with (output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    settings = {
        "arguments": vars(args),
        "checkpoint_path": str(checkpoint_dir),
        "baseline_model": MODEL_ID,
        "instruction": INSTRUCTION,
        "split": "development",
        "subset_seed": 42,
        "do_sample": False,
        "device": str(device),
    }

    with (output_dir / "evaluation_config.json").open("w") as file:
        json.dump(settings, file, indent=2)

    # Preserve every report, reference, and prediction for later error analysis
    with (output_dir / "predictions.jsonl").open("w") as file:
        for index, example in enumerate(examples):
            record = {
                "subset_index": index,
                "source": example["source"],
                "images_path": example["images_path"],
                "report": example["radiology_report"],
                "reference": example["layman_report"],
                "baseline": baseline_summaries[index],
                "trained": trained_summaries[index],
            }
            file.write(json.dumps(record) + "\n")

    print(json.dumps(metrics, indent=2), flush=True)
    print("Saved evaluation results to:", output_dir, flush=True)


if __name__ == "__main__":
    main()