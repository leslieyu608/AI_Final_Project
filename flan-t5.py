# pip install accelerate
from main import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration


def flan_t5(input_text, model_index):

    tokenizer = T5Tokenizer.from_pretrained(
        "google/flan-t5-" + model_index)
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-" + model_index, device_map="auto")

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, min_length=128,
                             max_new_tokens=512,
                             no_repeat_ngram_size=2,
                             early_stopping=True)
    print(tokenizer.decode(outputs[0]))
