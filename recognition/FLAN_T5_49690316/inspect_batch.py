from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from config import MODEL_ID, INSTRUCTION
from dataset import load_audited_data

import torch
from modules import RadiologySummarizer


"""
Prepare a batch containing two report-summary pairs. 
"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) # Defines the tokenizer
dataset, clean_train = load_audited_data() # Loads the data

features = [] # Creates a list to hold the two prepared examples

for example in clean_train.select(range(2)): # Selects rows at positions 0 and 1, processes them
    encoded = tokenizer(INSTRUCTION + example["radiology_report"])
    # Tokenizes the instructed input without padding


    target = tokenizer(text_target=example["layman_report"])
    # Tokenizes the reference summary separately

    encoded["labels"] = target["input_ids"]
    # Adds the expected output IDs under the key "labels". 
    
    features.append(encoded)
    # Adds prepared example to the list

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True, # Pads to the longest sequence in the batch
    label_pad_token_id=-100, # Marks target padding to be ignored by loss
    return_tensors="pt", # Produces pytorch tensors
)

batch = collator(features) # Combines the two examples into one batch

for name, tensor in batch.items():
    print(name, tensor.shape)

print("\nLabels:", batch["labels"])



"""
Now check that FLAN-T5 accepts the prepared batch and calculates a finite training loss.
Model will compare token predictions with reference labels.
"""


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
) # Select device

model = RadiologySummarizer(MODEL_ID).to(device) # Creates wrapper, loads flan-t5 in it, and moves to selected device
model.eval() # Loads model onto that device and activates eval mode

device_batch = {
    name: tensor.to(device)
    for name, tensor in batch.items()
} # Create a dictionary with same keys as batch, moving each tensor onto the models device

with torch.no_grad():
    outputs = model(**device_batch) # Disable gradient tracking

print("\nDevice:", device)
print("Batch loss:", outputs.loss.item())

assert torch.isfinite(outputs.loss).item(), "Loss is not finite" # Check that we didn't get NaN or infinity






model.train() # Activate training mode
model.zero_grad(set_to_none=True) # Clear any existing gradients

outputs = model(**device_batch) # Run another forward pass with gradient tracking enabled
outputs.loss.backward() # Calculate gradients

gradient = model.model.shared.weight.grad # Gets gradients for T5/s shared token-embedding matrix

print("\nBackward-pass loss:", outputs.loss.item())
print("Embedding gradient norm:", gradient.norm().item()) # Prints one number describing overall magnitude of that gradient tensor

assert torch.isfinite(gradient).all().item(), "Gradient contains nonfinite values"

model.zero_grad(set_to_none=True) # Clear diagnostic gradients afterward






"""
Now apply one optimiser update
Complete one training step: calculate predictions and loss, calculate graidents, then update model weights
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
# Creates AdamW optimiser with learning rate of 1e-4

tracked_parameter = model.model.encoder.final_layer_norm.weight # Selects small tensor of learned normalisation weights to inspect
before_update = tracked_parameter.detach().clone() # Saves an independent copy of its current values

optimizer.zero_grad(set_to_none=True) # Clear old gradients

outputs = model(**device_batch) 
outputs.loss.backward() # Calculates a fresh loss and its gradients
optimizer.step() # Use these gradients to update the model parameters

change = (
    tracked_parameter.detach() - before_update
).norm().item() # Measures magnitude of difference between updated values and old values

print("\nLoss before optimiser update:", outputs.loss.item())
print("Tracked parameter change:", change)

assert change > 0, "Tracked parameter did not change"

optimizer.zero_grad(set_to_none=True) # Reset old gradients again