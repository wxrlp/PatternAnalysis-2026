import ast

from datasets import load_dataset
from config import INSTRUCTION



DATASET_ID = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
# Stores the dataset name in case we need it later. 

def get_image_identifiers(example): 
    """Extract source-qualified image or session identifiers from one row.
    example will be a python dictionary representing one dataset row.
    For example:
    {
        "source": "PadChest",
        "images_path": "some_image.png",
        "radiology_report": "No abnormalities.",
        "layman_report": "The scan did not show any abnormalities."
    }
    """
    source = example["source"] # looks up the "source" key and assigns it's value, PadChest, to source
    raw_paths = example["images_path"]

    if source == "OpenI":
        paths = ast.literal_eval(raw_paths) # Converts OpenI's stored list into an actual list
    else:
        paths = [raw_paths] # ALl other sources get their identifiers put into a singleton list

    return {(source, path.strip()) for path in paths}


def load_audited_data():
    """
    Load the dataset and excludes training rows that share validation identifiers.
    Returns both the original splits and the filtered training data.
    Leaves validation and test set unchanged and doesn't create a development split yet.
    """
    dataset = load_dataset(DATASET_ID)
    # Loads the original splits, reusing cached downloads whenever possible

    validation_identifiers = set()

    for example in dataset["validation"]:
        validation_identifiers.update(get_image_identifiers(example)) 
        # Adds identifiers from each validation row and adds them to the set

    def keep_example(example):
        """Helper function that extracts the training row's identifiers and
        returns True when the two sets share no elements, meaning the row should be kept."""
        identifiers = get_image_identifiers(example)
        return identifiers.isdisjoint(validation_identifiers)

    clean_train = dataset["train"].filter(keep_example) # Applies that decision to every training row

    return dataset, clean_train # Returns two objects in a tuple, the original dataset and the filtered split




"""
Create a reusable function that converts a report-summary pair into model inputs and labels. 
"""

def tokenize_example(example, tokenizer):
    """Tokenise a complete report–summary pair without padding or truncation."""
    model_input = INSTRUCTION + example["radiology_report"]

    encoded = tokenizer(
        model_input,
        truncation=False,
        padding=False,
    ) # Produces input IDs and an attention mask, no padding or truncation

    target = tokenizer(
        text_target=example["layman_report"],
        truncation=False,
        padding=False,
    ) # Tokenises the layman summary separately, no padding or truncation

    encoded["labels"] = target["input_ids"]

    return encoded # Returns the prepared example