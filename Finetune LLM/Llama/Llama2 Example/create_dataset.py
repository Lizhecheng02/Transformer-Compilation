"""
The code below is used to create "guanaco-llama2-1k" dataset in huggingface,
which will be used for finetuning llama2 model.
"""

from datasets import load_dataset
import re

dataset = load_dataset("timdettmers/openassistant-guanaco")
print(dataset)

dataset = dataset["train"].shuffle(seed=42).select(range(5000))


def transform_conversation(example):
    conversation_text = example["text"]
    segments = conversation_text.split("###")

    reformatted_segments = []
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace("Human:", "").strip()
        if (i + 1) < len(segments):
            assistant_text = segments[i + 1].strip().\
                replace("Assistant:", "").strip()
            reformatted_segments.append(
                f"<s>[INST] {human_text} [/INST] {assistant_text} </s>")
        else:
            reformatted_segments.append(f"<s>[INST] {human_text} [/INST] </s>")

    return {"text": "".join(reformatted_segments)}


transformed_dataset = dataset.map(transform_conversation)
