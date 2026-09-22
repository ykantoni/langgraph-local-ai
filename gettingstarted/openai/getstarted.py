import os
from openai import OpenAI

# Set HF_TOKEN in your environment before running, e.g.:
#   PowerShell: $env:HF_TOKEN="hf_..."
#   cmd:        set HF_TOKEN=hf_...

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