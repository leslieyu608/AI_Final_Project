import importlib

flan_T5 = importlib.import_module("flan-t5")
flan_alpaca = importlib.import_module("flan-alpaca")

def main():
    input_text = "Write a fantasy story about a tragic knight in an apocalyptic world."
    flan_T5.model(input_text, model_index=0)
    flan_alpaca.model(input_text, model_index=0)


if __name__ == "__main__":
    main()
