from models.roformer import RoFormerForCausalLM
from transformers import AutoTokenizer
import torch
from huggingface_hub import HfApi, create_repo
import os


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

checkpoint_path = "/home/ubuntu/roformer"
save_path = "/home/ubuntu/roformer/roformer-rope-disabled"

# Load the trained model
model = RoFormerForCausalLM.from_pretrained(checkpoint_path)
model = model.to(device)

# Save the model locally first
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Upload to Hugging Face Hub
# Replace these with your Hugging Face credentials
HF_USERNAME = "chrisjob1021"  # Replace with your Hugging Face username
MODEL_NAME = "roformer-rope-disabled"  # Name for your model on Hugging Face
REPO_NAME = f"{HF_USERNAME}/{MODEL_NAME}"

# Create a new repository on Hugging Face Hub
try:
    create_repo(REPO_NAME, private=False)  # Set private=True if you want a private repository
except Exception as e:
    print(f"Repository might already exist: {e}")

# Initialize the Hugging Face API
api = HfApi()

# Upload the model files
print(f"Uploading model to {REPO_NAME}...")
api.upload_folder(
    folder_path=save_path,
    repo_id=REPO_NAME,
    repo_type="model"
)

print(f"Model successfully uploaded to https://huggingface.co/{REPO_NAME}")



