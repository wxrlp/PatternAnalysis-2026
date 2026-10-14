from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import MODEL_ID, INSTRUCTION
from dataset import load_audited_data, split_train_development
from modules import RadiologySummarizer

import nltk
import numpy as np
from rouge_score import rouge_scorer


def generate_summaries(model_path, reports, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RadiologySummarizer(str(model_path)).to(device) 
    model.eval() # Define tokenizer, model, and set to evaluation mode

    summaries = []

    for report in reports: # Processes one report at a time
        inputs = tokenizer(
            INSTRUCTION + report,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        summaries.append(
            tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ) # Decodes each output and adds it to the result
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
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    _, clean_train = load_audited_data()
    _, development = split_train_development(clean_train) # Filter data and split into training/development

    examples = development.shuffle(seed=42).select(range(32)) # Selects first 5 examples
    reports = examples["radiology_report"]

    checkpoint_dir = (
        Path(__file__).resolve().parent / "checkpoints" / "pilot_lora_lr001"
    )

    baseline_summaries = generate_summaries(MODEL_ID, reports, device)
    pilot_summaries = generate_summaries(checkpoint_dir, reports, device)
    # Generates from the original and trained checkpoints using the same reports


    references = examples["layman_report"]

    print("\nBaseline ROUGE:", calculate_rouge(
        baseline_summaries, references
    ))
    print("Pilot ROUGE:", calculate_rouge(
        pilot_summaries, references
    ))


    for index, example in enumerate(examples.select(range(2))):
        print(f"\nExample {index + 1} | Source: {example['source']}")
        print("Report:", example["radiology_report"])
        print("Reference:", example["layman_report"])
        print("Baseline:", baseline_summaries[index])
        print("Pilot:", pilot_summaries[index])


if __name__ == "__main__":
    main()