from datasets import load_dataset

dataset = load_dataset("HuggingFaceTB/smoltalk", 'all')

def convert_to_chatml(example):
    return {
        "messages": [
            {"role": "user", "content": example["content"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


dataset = dataset.map(convert_to_chatml, batched=True)
print(dataset["train"][0])


