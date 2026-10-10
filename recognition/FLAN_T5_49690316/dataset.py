import ast

from datasets import load_dataset



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