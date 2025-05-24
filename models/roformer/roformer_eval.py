from models.roformer import RoFormerForCausalLM
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import os
import json
from utils import Config
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the given dataloader."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            batch_size = input_ids.size(0)
            attention_mask = batch["attention_mask"].to(device)
            
            # Create labels by cloning input_ids
            # Model will handle the shift internally
            labels = input_ids.clone()
            
            # Forward pass with labels
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            
            # Get predictions
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            
            # Store predictions and labels (excluding padding and ignored tokens)
            # The sequence of operations is:
            # valid_mask filters out padding tokens and ignored tokens (where label is -100)
            valid_mask = (labels != -100) & (attention_mask == 1)
            # .cpu() moves the tensor from GPU to CPU
            # .numpy() converts the PyTorch tensor to a NumPy array
            # extend() adds these values to the running list of all predictions/labels
            all_predictions.extend(predictions[valid_mask].cpu().numpy())
            # NumPy is required in this case because the code is using scikit-learn's accuracy_score function to calculate the model's accuracy.
            # NumPy arrays can only be created from CPU tensors, not GPU tensors
            all_labels.extend(labels[valid_mask].cpu().numpy())
            
            # Accumulate loss and count tokens
            # This creates a boolean tensor "labels" where each element is True if the label is not -100, False if it is -100
            # .sum() sums up all the True values in the boolean tensor
            # In PyTorch, True is treated as 1 and False as 0 when summing
            # So this gives us the count of valid tokens (non-padding tokens)
            total_loss += loss.item() * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "total_tokens": total_tokens
    }

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, device=None):
    """Generate text from the model."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    generated_ids = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated_ids)
            logits = outputs["logits"]
            
            # Get the last token's probabilities
            # First dimension : - represents all batches
            # Second dimension -1 - represents the last position in the sequence
            # Third dimension : - represents all possible token probabilities
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
            # This sets all logits to negative infinity
            next_token_logits[0, :] = float('-inf')
            # Restore only the top k logits
            next_token_logits[0, top_k_indices[0]] = top_k_logits[0]
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids[0])

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model from HuggingFace
    model_name = "chrisjob1021/RoFormer-base-124M-RoPE-disabled"
    print(f"Loading model from {model_name}")
    
    # Load model
    model = RoFormerForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    ds = load_dataset("chrisjob1021/gpt2_tokenized_concatenated_openwebtext")
    eval_dataset = ds["train"].train_test_split(test_size=0.0001, seed=42)["test"]
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, eval_dataloader, device)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total tokens evaluated: {metrics['total_tokens']}")
    
    # Generate some example texts
    print("\nGenerating example texts:")
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In the beginning",
        "The future of artificial intelligence"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated_text = generate_text(model, tokenizer, prompt, device=device)
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 