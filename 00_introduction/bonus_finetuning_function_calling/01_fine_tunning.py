from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# Load dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)


# # Initialize trainer
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     processing_class=tokenizer,
# )

# # Start training
# trainer.train()


# Configure packing
training_args = SFTConfig(packing=True)

def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args, formatting_func=formatting_func)

trainer.train()