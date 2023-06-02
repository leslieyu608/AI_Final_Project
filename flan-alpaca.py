from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def model(input_text, model_index):
    model_size = ["base", "large", "xl", "gpt4-xl"]

    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-" + model_size[model_index])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "declare-lab/flan-alpaca-" + model_size[model_index], device_map="auto"
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        min_length=300,
        max_new_tokens=1024,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
