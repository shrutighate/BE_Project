# from safetensors import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartTokenizer, BartForConditionalGeneration
import torch

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


def summarize(input_text, max_summary_length=150):

    # ------------------------------------------
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    #
    # inputs = tokenizer(input_text, max_length=4096, return_tensors='pt', truncation=True)
    # summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=50, max_length=150, early_stopping=True)
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    #
    # return summary
    # --------------------------------
    chunk_size = 4096
    chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]

    # Generate summary for each chunk
    summaries = []
    for chunk in chunks:
        # Tokenize input text
        inputs = tokenizer(chunk, max_length=chunk_size, return_tensors='pt', truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=50, max_length=max_summary_length ,
                                     early_stopping=True)

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Join summaries of chunks into a single summary
    final_summary = " ".join(summaries)

    return final_summary
