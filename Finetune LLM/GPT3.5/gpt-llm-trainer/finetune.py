"""
First, we need to generation our finetuning data using gpt model.
"""

import pandas as pd
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import random
import openai
import os

prompt = "A model that takes in a puzzle-like reasoning-heavy question in English, and responds with a well-reasoned, step-by-step thought out response in Spanish."
temperature = 0.5

"""
The minimum number of examples to finetune gpt3.5 model is about 20.
"""
number_of_examples = 20

openai.api_key = "Enter your api key here"

N_RETRIES = 3

"""
Using gpt-4 model to generate examples.
"""


@retry(stop=stop_after_attempt(N_RETRIES),
       wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=0.5):
    messages = [{
        "role": "system",
        "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
    }]

    if len(prev_examples) > 0:
        if len(prev_examples) > 3:
            prev_examples = random.sample(prev_examples, 8)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1024
    )

    return response.choices[0].message["content"]


prev_examples = []
for i in range(number_of_examples):
    print(f"Generating example {i}")
    example = generate_example(
        prompt=prompt,
        prev_examples=prev_examples,
        temperature=temperature
    )
    prev_examples.append(example)

print(prev_examples)

"""
Using gpt-4 model to generate system message.
"""


def generate_system_message(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
            },
            {
                "role": "user",
                "content": prompt.strip()
            }
        ],
        temperature=temperature,
        max_tokens=512
    )

    return response.choices[0].message["content"]


system_message = generate_system_message(prompt)
print(
    f"The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result."
)

"""
Clear our examples got from gpt model, first format them into a dataframe, and then convert them to jsonl format which is fit for finetuning gpt-3.5-turbo.
"""

prompts = []
responses = []

for example in prev_examples:
    try:
        split_example = example.split("-----------")
        prompts.append(split_example[1].strip())
        responses.append(split_example[3].strip())
    except:
        pass

df = pd.DataFrame({
    "prompt": prompts,
    "response": responses
})
df = df.drop_duplicates()

print("There are " + str(len(df)) + " successfully-generated examples.")

training_examples = []

for index, row in df.iterrows():
    training_example = {
        "message": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}
        ]
    }
    training_examples.append(training_example)

with open("./training_examples.jsonl", "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

"""
Normal steps to finetune gpt-3.5 using api.
"""

file_id = openai.File.create(
    file=open("./training_examples.jsonl", "rb"),
    purpose="fine-tune"
)["id"]

job_id = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)["id"]

openai.FineTuningJob.list_events(id=job_id, limit=10)

"""
Once model is trained, we need to grab the fine-tuned model name.
"""

model_name_pre_object = openai.FineTuningJob.retrieve(job_id)
model_name = model_name_pre_object.fine_tuned_model
print(model_name)

"""
Try our fine-tuned model.
"""
response = openai.Completion.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": df["prompt"].sample().values[0]
        }
    ]
)

print(response.choices[0].message["content"])
