# backend/core/llm.py
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def embed(texts, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def chat(messages, model="gpt-4o-mini", temperature=0.2):
    return client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    ).choices[0].message.content
