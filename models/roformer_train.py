from datasets import load_dataset
from transformers import AutoTokenizer
from utils import TrainingConfig, Config
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or your custom one
tokenizer.pad_token = tokenizer.eos_token

training_config = TrainingConfig()
config = Config(vocab_size=tokenizer.vocab_size,
d_model=768, num_heads=12, ffn_dim=3072,
num_layers=12, max_seq_len=tokenizer.model_max_length )

savepath = "/home/ubuntu/roformer-base"

# Create a config dictionary for the model
config_dict = {k: getattr(config, k) for k in vars(config) 
            if not k.startswith('_') and not callable(getattr(config, k))}

# Save the config as JSON
import os
import json

# Create directory if it doesn't exist
if not os.path.exists(savepath):
    os.makedirs(savepath)
    
# Save the config file
config_path = os.path.join(savepath, "config.json")
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("chrisjob1021/gpt2_tokenized_concatenated_openwebtext")

# Split the dataset into training and evaluation sets
train_test_split = lm_dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

tokenizer.pad_token = tokenizer.eos_token

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback
from models import RoFormerForCausalLM, RoFormerDecoder
import torch

if True:
    # model initialization
    model_base = RoFormerDecoder(config)
    model = RoFormerForCausalLM(model_base, config)
if False: model = RoFormerForCausalLM.from_pretrained(savepath)
model = model.to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

import os
# Get the absolute path for logs
log_dir = os.path.join(os.path.dirname(savepath), "logs")
# Create the logging directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

args = TrainingArguments(
    output_dir=savepath,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8, # Accumulate gradients over N steps
    #With gradient accumulation (gradient_accumulation_steps=8):
        # You split what would have been one batch into 8 smaller micro-batches
        # For each micro-batch, you:
        # Load 1/8th of the data into memory
        # Do a forward pass (storing 1/8th of the activations)
        # Do a backward pass (computing 1/8th of the gradients)
        # ACCUMULATE the gradients (don't update weights yet)
        # Clear the activations (but keep gradients)

    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,

    eval_steps=100,
    eval_strategy="steps",
    eval_accumulation_steps=8,
    per_device_eval_batch_size=16,

    warmup_steps=100,
    logging_dir=log_dir,
    logging_steps=100,
    save_steps=100,
    save_total_limit=10,
    save_strategy="steps",
    save_safetensors=False,
    # report_to="tensorboard",
    gradient_checkpointing=False,

    #With Gradient Checkpointing:
        # During the forward pass, only store activations at certain "checkpoints"
        # During backpropagation, RECOMPUTE the intermediate activations as needed
        # This means doing some forward computations twice, but using much less memory
    # Without checkpointing, you need to store activations for all 12 layers. With checkpointing, you might only store activations every few layers and recompute the rest during backprop.
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    ]
)

trainer.train()