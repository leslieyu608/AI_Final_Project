# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_size = ["base", "small", "large", "xl"]

tokenizer = T5Tokenizer.from_pretrained(
    "google/flan-t5-" + model_size[3])
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-" + model_size[3], device_map="auto")

input_text = "Write a fantasy story about a tragic knight in an apocalyptic world."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids, min_length=128,
                         max_new_tokens=512,
                         no_repeat_ngram_size=2,
                         early_stopping=True)
print(tokenizer.decode(outputs[0]))
