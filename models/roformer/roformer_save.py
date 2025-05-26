from models.roformer import RoFormerForCausalLM
from transformers import AutoTokenizer
import torch
import shutil
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

checkpoint_path = "/home/ubuntu/roformer/rope-enabled/checkpoint-20000"
save_path = "/home/ubuntu/roformer/RoFormer-base-124M-RoPE-enabled"

# Copy config.json from checkpoint to save path
config_source = os.path.join(os.path.dirname(checkpoint_path), "config.json")
config_dest = os.path.join(checkpoint_path, "config.json")
shutil.copy2(config_source, config_dest)

# Load the trained model
model = RoFormerForCausalLM.from_pretrained(checkpoint_path)
model = model.to(device)

# Save the model locally first
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)