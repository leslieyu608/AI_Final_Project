import importlib

flan_T5 = importlib.import_module("flan-t5")
flan_alpaca = importlib.import_module("flan-alpaca")


def main():
    input_text = "Write a fantasy story about a tragic knight in an apocalyptic world."

    print("flan-t5-base:")
    flan_T5.model(input_text, model_index=0)

    print("flan-t5-small:")
    flan_T5.model(input_text, model_index=1)

    print("flan-t5-large:")
    flan_T5.model(input_text, model_index=2)

    print("flan-t5-xl:")
    flan_T5.model(input_text, model_index=3)

    print("flan-alpaca-base:")
    flan_alpaca.model(input_text, model_index=0)

    print("flan-alpaca-large:")
    flan_alpaca.model(input_text, model_index=1)

    print("flan-alpaca-xl:")
    flan_alpaca.model(input_text, model_index=2)

    print("flan-alpaca-gpt4-xl:")
    flan_alpaca.model(input_text, model_index=3)


if __name__ == "__main__":
    main()
