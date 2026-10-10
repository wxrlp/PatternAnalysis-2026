
from collections import Counter
import ast

from dataset import get_image_identifiers, load_audited_data # Import from our own dataset.py file. 


dataset, clean_train = load_audited_data()

print(dataset)

for split_name, split in dataset.items():
    print(f"\nSplit: {split_name}")
    print(f"Rows: {len(split)}")
    print(f"Columns: {split.column_names}")


example = dataset["train"][0] 
# Selects training split and then first example from training split

print("\nSource:", example["source"])
print("\nRadiology report:")
print(example["radiology_report"])

print("\nPlain-language reference:")
print(example["layman_report"])

for split_name, split in dataset.items(): # Splits between training, validation, and test data
    for column in ["radiology_report", "layman_report"]: # Check the missing cuonts in the radiology report and the laymans report
        # Checks radiology report in training, validation, test data, then does the same with layman report
        missing_count = sum(1 for text in split[column] if text is None or not text.strip())
        # Goes through every value in current column and counts for problematic entries
        # If entry is None (empty) or not text.strip() (empty when whitespace is stripped away), then add 1 to the count

        print(f"{split_name} | {column} | missing or blank: {missing_count}") 
        # No missing values except for layman report in test data, because that's what we will be generating


report_sets = {}

for split_name, split in dataset.items():
    report_sets[split_name] = {text.strip() for text in split["radiology_report"]}
    # builds a set of reports for that split, for example: {'train': xxx, 'validation': yyy, 'test': zzz}

for first, second in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    overlap = report_sets[first] & report_sets[second]
    print(f"{first} / {second} | shared reports: {len(overlap)}")


image_sets = {}

for split_name, split in dataset.items():
    image_sets[split_name] = {(source, path) for source, path in zip(split["source"], split["images_path"]) if path}
    # Creates a tuple identifying each image by both source and path, since different datasets can reuse filenames. 
    # Excludes empty or missing paths

for first, second in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    overlap = image_sets[first] & image_sets[second]
    print(f"{first} / {second} | shared image identifiers: {len(overlap)}") # Finds intersections shared by sets. Test sets shares nothing with training and validation


validation_reports = report_sets["validation"]
validation_images = image_sets["validation"]

overlapping_rows = 0

for example in dataset["train"]:
    report = example["radiology_report"].strip()
    image_id = (example["source"], example["images_path"])

    if report in validation_reports or image_id in validation_images:
        overlapping_rows += 1

print("Training rows matching validation:", overlapping_rows)
print("Training rows remaining:", len(dataset["train"]) - overlapping_rows)





training_report_counts = Counter(
    text.strip() for text in dataset["train"]["radiology_report"]
)

shared_reports = report_sets["train"] & report_sets["validation"]

most_common_shared = sorted(
    shared_reports,
    key=lambda report: training_report_counts[report],
    reverse=True,
)[:5] # Sorts those text by training occurence counts, highest first, and selects 5. 


for report in most_common_shared:
    print("\nTraining occurrences:", training_report_counts[report])
    print("Report:", report)





image_overlap_rows = 0

for example in dataset["train"]:
    image_id = (example["source"], example["images_path"])

    if image_id in image_sets["validation"]:
        image_overlap_rows += 1

print("Training rows with shared image identifiers:", image_overlap_rows)


"""
Now define a function that can extract identifiers from any row. 
OpenI needs its list to be parsed, but the other sources already provide 
one identifier per row.
"""




openi_example = next(
    example for example in dataset["train"]
    if example["source"] == "OpenI"
) # Finds the first OpenI row.

raw_paths = openi_example["images_path"] # Gets the field as it's stored in the dataset: a string
parsed_paths = ast.literal_eval(raw_paths) # Converts the string into a list containing the individual paths

print("Before parsing:", type(raw_paths))
print("After parsing:", type(parsed_paths))
print("Individual paths:", parsed_paths)

print(get_image_identifiers(openi_example))


"""
Now repeat the overlap check using individual image paths.
The earlier check compared OpenI's entire list as a string, so it could miss
a shared image appearing in two different lists.
"""


individual_image_sets = {}

for split_name, split in dataset.items(): # visit each original dataset split
    identifiers = set()  # start an empty set for the current split

    for example in split:
        identifiers.update(get_image_identifiers(example))
        # Visit every row in the current split and extract the row identifiers and adds
        # them to the set. Duplicates are automatically ignored. 
        # Update is used instead of .add() because .add() only adds a single element. 

    individual_image_sets[split_name] = identifiers

for first, second in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    overlap = individual_image_sets[first] & individual_image_sets[second]
    print(f"{first} / {second} | shared individual identifiers: {len(overlap)}")



print("Original training rows:", len(dataset["train"]))
print("Filtered training rows:", len(clean_train))



"""
We need to remove overlap between traininig and validation set. 
There are different kinds of overlap:
    Same image identifiers are probably the exact same image and need to be removed.
    Same report wording can occur for different patioents, especially phrases like "no abnormalities". We can't remove this.
    Same patient, different scans: we need to check this
"""


# This earlier audit counts whole images_path fields, not individual OpenI paths.
# Keep it for inspection; it is not the final development-split grouping rule.
clean_image_counts = Counter(
    (example["source"], example["images_path"])
    for example in clean_train
    if example["images_path"]
) # If a path exists, count the number of source-path pairs in the training set

repeated_identifiers = sum(
    1 for count in clean_image_counts.values() if count > 1
) # Counts how many distinct identifiers occur more than once

print("Repeated image identifiers in clean training:", repeated_identifiers)



# Count number of missing image paths
missing_image_paths = len(clean_train) - sum(clean_image_counts.values()) 
print("Rows with missing or empty image paths:", missing_image_paths)



"""
Now build a list containing each image identifier exactly once. 
We'll later split the list then assign all rows belonging to an identfiier to the same split
"""


image_groups = sorted(clean_image_counts.keys())

print("Training rows:", len(clean_train))
print("Unique image groups:", len(image_groups))
print("First three groups:", image_groups[:3])



"""
Show some identifiers from every source. 
"""

sources = sorted(set(clean_train["source"])) # Remove duplicates and sort the source names

for source in sources:
    source_groups = [path for group_source, path in image_groups if group_source == source]

    print(f"\nSource: {source}")
    print("Unique groups:", len(source_groups))
    print("Example identifiers:", source_groups[:5])



"""
Now confirm that no individual image identifier in clean_train appears in the validation set.
"""
clean_identifiers = set()

for example in clean_train: # visit each retained row, extract identifiers, and add to set
    clean_identifiers.update(get_image_identifiers(example))

remaining_overlap = (
    clean_identifiers & individual_image_sets["validation"]
)

print("Remaining training/validation identifier overlap:", len(remaining_overlap))

assert not remaining_overlap, "Training and validation still share identifiers"