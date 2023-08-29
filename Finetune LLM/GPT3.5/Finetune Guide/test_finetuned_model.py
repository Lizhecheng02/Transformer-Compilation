import langchain
import openai
import pinecone
import requests
from time import sleep
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType, initialize_agent
from chains import VectorDBChain

openai.api_key = ""

job_id = ""
"""
Get your finetuned model Id after running finetune.py
"""

# print(openai.FineTuningJob.retrieve(job_id))

while True:
    res = openai.FineTuningJob.retrieve(job_id)
    if res["finished_at"] != None:
        break
    else:
        print("Please waiting ...")
        sleep(100)

print(res)

finetuned_model = res["fine_tuned_model"]
print(finetuned_model)

llm = ChatOpenAI(
    temperature=0.2,
    model_name=finetuned_model
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)

vdb = VectorDBChain(
    index_name="Enter your index name",
    environment="Enter your environment name",
    pinecone_api_key="Enter your pinecone api key"
)

vdb_tool = Tool(
    name=vdb.name,
    func=vdb.query,
    description="This tool allows you to get research information about LLMs"
)

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[vdb_tool],
    llm=llm,
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False
)

agent("Tell me about Llama2 Model")
