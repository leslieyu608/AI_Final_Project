from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    input_text = "Write a fantasy story about a tragic knight in an apocalyptic world."
    model_index = ["base", "small", "large", "xl"]
    flan_t5(input_text, model_index[0])


if __name__ == "__main__":
    main()
