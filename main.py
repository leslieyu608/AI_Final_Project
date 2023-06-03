from tqdm import tqdm
import importlib

flan_T5 = importlib.import_module("flan-t5")
flan_alpaca = importlib.import_module("flan-alpaca")


def main():
    # input_text = "Write a fantasy story about a tragic knight in an apocalyptic world."
    # input_text = "Write an adventure story in a magic world."
    input_text = "Write a righteous story about a group of superheroes who ultimately destroy the super villain"

    print("flan-t5-small:")
    for i in tqdm(range(1)):
        flan_T5.model(input_text, model_index=0)

    print("flan-t5-base:")
    for i in tqdm(range(1)):
        flan_T5.model(input_text, model_index=1)

    print("flan-t5-large:")
    for i in tqdm(range(1)):
        flan_T5.model(input_text, model_index=2)

    print("flan-t5-xl:")
    for i in tqdm(range(1)):
        flan_T5.model(input_text, model_index=3)

    print("flan-alpaca-base:")
    for i in tqdm(range(1)):
        flan_alpaca.model(input_text, model_index=0)

    print("flan-alpaca-large:")
    for i in tqdm(range(1)):
        flan_alpaca.model(input_text, model_index=1)

    print("flan-alpaca-xl:")
    for i in tqdm(range(1)):
        flan_alpaca.model(input_text, model_index=2)

    print("flan-alpaca-gpt4-xl:")
    for i in tqdm(range(1)):
        flan_alpaca.model(input_text, model_index=3)


if __name__ == "__main__":
    main()
