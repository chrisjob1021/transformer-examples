from huggingface_hub import create_repo, HfApi

# Upload to Hugging Face Hub
# Replace these with your Hugging Face credentials
HF_USERNAME = "chrisjob1021"  # Replace with your Hugging Face username
MODEL_NAME = "RoFormer-base-124M-RoPE-disabled"  # Name for your model on Hugging Face
REPO_NAME = f"{HF_USERNAME}/{MODEL_NAME}"
save_path = "/home/ubuntu/roformer/RoFormer-base-124M-RoPE-disabled"

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
