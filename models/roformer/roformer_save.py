from models.roformer import RoFormerForCausalLM
from transformers import AutoTokenizer
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

checkpoint_path = "/home/ubuntu/roformer/checkpoint-18500"
save_path = "/home/ubuntu/roformer/RoFormer-base-124M-RoPE-disabled"

# Load the trained model
model = RoFormerForCausalLM.from_pretrained(checkpoint_path)
model = model.to(device)

# Save the model locally first
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)