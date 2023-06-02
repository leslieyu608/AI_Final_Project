from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def model(input_text, model_index):
    model_size = ["base", "large", "xl", "gpt4-xl"]

    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-" + model_size[model_index])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "declare-lab/flan-alpaca-" + model_size[model_index], device_map="auto", offload_folder="offload"
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature = 0.8,
        top_p = 0.95,
        top_k = 50,
        do_sample = True,
    )

    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
