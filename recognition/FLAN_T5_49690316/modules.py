import torch
from transformers import AutoModelForSeq2SeqLM


class RadiologySummarizer(torch.nn.Module): 
    # Defines class as a pytorch module, gives it standard features such as .to(device), .train(), .eval(), etc
    """Wrap pretrained FLAN-T5 for radiology summarisation."""

    def __init__(self, model_id): # model_id specifies the checkpoint
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id) # loads FLAN-T5 and stores it inside the wrapper

    def forward(self, input_ids, attention_mask=None, labels=None): # Defines forward computation
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )



    def generate(self, input_ids, attention_mask=None, **generation_options):
        """
        Generate output token IDs using the wrapped FLAN-T5 model.
        **generation_options collects extra arguments into a dictionary. For example:
        "max_new_tokens=128, do_sample=False" becomes {"max_new_tokens": 128, "do_sample": False}
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_options,
        )




