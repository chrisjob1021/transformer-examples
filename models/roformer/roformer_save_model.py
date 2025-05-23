from transformers import AutoTokenizer
from models.roformer import RoFormerForCausalLM
from utils import Config
import torch

ckpt_dir = "/home/ubuntu/roformer/checkpoint-18500"   # your best step
save_dir = "/home/ubuntu/roformer/roformer-final"     # clean export folder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2").to(device)

# Load the trained model
model = RoFormerForCausalLM.from_pretrained(ckpt_dir).to(device)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir) 