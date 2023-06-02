from transformers import T5Tokenizer, T5ForConditionalGeneration


def model(input_text, model_index):
    model_size = ["base", "small", "large", "xl"]

    tokenizer = T5Tokenizer.from_pretrained(
        "google/flan-t5-" + model_size[model_index])
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-" + model_size[model_index])

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        min_length=128,
        max_new_tokens=512,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    print(tokenizer.decode(outputs[0]))
