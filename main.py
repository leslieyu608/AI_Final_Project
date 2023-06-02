from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    input_text = "Write a fantasy story about a tragic knight in an apocalyptic world."
    t5_base(input_text)


def t5_base(input_text):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs, min_length=128, max_new_tokens=512, no_repeat_ngram_size=2, early_stopping=True
    )

    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


if __name__ == "__main__":
    main()
