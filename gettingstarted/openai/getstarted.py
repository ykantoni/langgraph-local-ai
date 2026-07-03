import os
from openai import OpenAI

# $env:HF_TOKEN="<REDACTED_OPENAI_API_KEY>"
# 

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "user",
            "content": "Create a recipe for a chocolate cake."
        }
    ],
)

print(completion.choices[0].message)