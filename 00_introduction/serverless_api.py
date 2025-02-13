import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()
# os.environ["HF_TOKEN"]=os.getenv("HF_API_KEY")

RUNTIME = "HF"
RUNTIME = "local"

# Hugging Face hosted endpoint
# client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")

# Local LM Studio endpoint (default port is 1234)
# client = InferenceClient("http://localhost:1234/v1")

output = client.text_generation(
    "The capital of France is",
    max_new_tokens=100,
)

print("\nRaw Prompt Output:")
print(output)


prompt=(
    """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    The capital of france is<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
)

output = client.text_generation(
    prompt,
    max_new_tokens=100,
)

print("\nChat Template Output:")
print(output)


output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of france is"},
    ],
    stream=False,
    max_tokens=1024,
)

print("\nChat API Output:")
print(output.choices[0].message.content)