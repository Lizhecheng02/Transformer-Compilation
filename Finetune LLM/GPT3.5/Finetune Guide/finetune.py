"""
In order to finetune GPT3.5-turbo, we need to use jsonl format file.
Here is an example file content:
[
  {
    "messages": [
      { "role": "system", "content": "You are an assistant that crafts compelling product descriptions for Amazon listings." },
      { "role": "user", "content": "Describe a wireless Bluetooth speaker." },
      { "role": "assistant", "content": "Experience crystal-clear audio with this cutting-edge wireless Bluetooth speaker. Designed for music enthusiasts, its sleek design and powerful bass ensure an immersive audio experience. Perfect for on-the-go tunes!" }
    ]
  },
  {
    "messages": [
      { "role": "system", "content": "You are an assistant that crafts compelling product descriptions for Amazon listings." },
      { "role": "user", "content": "Describe a digital camera." },
      { "role": "assistant", "content": "Capture life's precious moments with our state-of-the-art digital camera. Boasting high-resolution capabilities, intuitive controls, and compact design, it's the ultimate tool for photography enthusiasts." }
    ]
  }
]
"""

import openai


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def save_file(filepath, content):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content)


openai.api_key = "Enter you api-key here"

with open("Enter your jsonl file path here", "rb") as file:
    response = openai.File.create(
        file=file,
        purpose="fine-tune"
    )

file_id = response["id"]
print("File uploaded successfully with ID:", file_id)

"""
Here we will get a file ID, which will be used for finetuning on this specific file.
Now we need to create a FineTuningJob agent to automatically.
Use openai >= 0.27.9, otherwise we cannot load FineTuningJob API
"""

model_name = "gpt-3.5-turbo"

agent = openai.FineTuningJob.create(
    training_file=file_id,
    model=model_name
)

job_id = response["id"]
print("Finetuning job created successfully with ID:", job_id)

"""
Now we can use our finetuned model on OPENAI Website Playground !!!
"""
