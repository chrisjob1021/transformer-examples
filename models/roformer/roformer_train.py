from datasets import load_dataset
from transformers import AutoTokenizer
from utils import TrainingConfig, Config
import torch
import os
import json
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback
from models.roformer import RoFormerForCausalLM, RoFormerDecoder
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.algorithms.ddp_comm_hooks")

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or your custom one
tokenizer.pad_token = tokenizer.eos_token

training_config = TrainingConfig()
config = Config(vocab_size=tokenizer.vocab_size,
    d_model=768, num_heads=12, ffn_dim=3072,
    num_layers=12, max_seq_len=tokenizer.model_max_length )

# Get the save path from the accelerate CLI arguments or use default
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="/home/ubuntu/roformer", 
                    help="Directory to save model checkpoints")
args, _ = parser.parse_known_args()
savepath = args.output_dir

# Create a config dictionary for the model
config_dict = {k: getattr(config, k) for k in vars(config) 
            if not k.startswith('_') and not callable(getattr(config, k))}

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

ds = load_dataset("chrisjob1021/gpt2_tokenized_concatenated_openwebtext")

# Split the dataset into training and evaluation sets
train_test_split = ds["train"].train_test_split(test_size=0.01, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

tokenizer.pad_token = tokenizer.eos_token

# Add command line argument for resuming from checkpoint
parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
accelerate_args, _ = parser.parse_known_args()

# if accelerate_args.resume and os.path.exists(savepath):
#     print(f"Resuming from checkpoint: {savepath}")
#     model = RoFormerForCausalLM.from_pretrained(savepath)
# else:
#     print("Initializing new model")
model_base = RoFormerDecoder(config)
model = RoFormerForCausalLM(model_base, config)
model = model.to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Get the absolute path for logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(os.path.dirname(savepath), "logs", f"run_{timestamp}")
# Create the logging directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

args = TrainingArguments(
    output_dir=savepath,

    learning_rate=6e-4,
    # lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.1,  # 10% of total training steps for warmup
    # Specify AdamW optimizer
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=0.5,

    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16, # Accumulate gradients over N steps
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

    eval_steps=500,
    eval_strategy="steps",
    eval_accumulation_steps=16,
    per_device_eval_batch_size=4,

    logging_dir=log_dir,
    logging_steps=50,

    save_steps=500,
    save_total_limit=10,
    save_strategy="steps",
    save_safetensors=False,

    ddp_find_unused_parameters=False,

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

# Resume from last checkpoint if available
if accelerate_args.resume:
    print(f"Resuming from checkpoint")
    trainer.train(resume_from_checkpoint=True)
else:
    print("Starting new training")
    trainer.train()
