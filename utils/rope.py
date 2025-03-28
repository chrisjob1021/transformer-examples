"""
Rotary Position Embedding (RoPE) implementation.

This module provides an implementation of the Rotary Position Embedding (RoPE) as described in:
"RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def apply_rope(x, past_seq_len=0, freq=10000.0, visualize=False, debug=False):
    # 1) Unpack the shape of the input
    batch_size, n_heads, seq_len, per_head_dim = x.shape
    assert per_head_dim % 2 == 0, "Head dimension must be even for pairwise RoPE." 

    # 2) Create a position index [0, 1, 2, ..., seq_len-1]
    #    shape: (seq_len,)
    positions = torch.arange(start=past_seq_len, end=past_seq_len+seq_len)
    if debug:
        print("positions.shape", positions.shape)

    # 3) Create an index over the half-dimension. We treat each dimension i as paired (2i, 2i+1).
    #    shape: (per_head_dim//2,)
    dim_idx = torch.arange(per_head_dim // 2)
    if debug:
        print("dim_idx.shape", dim_idx.shape)

    # 4) Compute the "angle" or "theta" for each pair of dimensions:
    #    RoPE defines these angles as positions * (base_freq ^ (-2*i / d))
    #    where d is the total head dimension. We exponentiate freq in the negative direction.
    #    shape: (seq_len, per_head_dim//2)
    theta = positions.unsqueeze(1) * (freq ** (-2 * dim_idx / per_head_dim))
    if debug:
        print("theta.shape", theta.shape)

    # 5) Compute sin and cos for all positions and dimensions
    #    shape: both are (seq_len, per_head_dim//2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # 6) Reshape cos/sin to allow broadcasting across (batch_size, n_heads)
    #    We want them to match the shape (batch_size, n_heads, seq_len, per_head_dim//2)
    #    when we broadcast. So we add dimensions:
    #    from (seq_len, per_head_dim//2) -> (1, 1, seq_len, per_head_dim//2)
    cos_theta = cos_theta.unsqueeze(0).unsqueeze(1)
    sin_theta = sin_theta.unsqueeze(0).unsqueeze(1)
    if debug:
        print("cos_theta.shape", cos_theta.shape)
        print("sin_theta.shape", sin_theta.shape)

    # 7) Reshape x so we can work with the final dimension as pairs: (..., 2)
    #    shape: (batch_size, n_heads, seq_len, per_head_dim//2, 2)
    x_reshaped = x.view(batch_size, n_heads, seq_len, per_head_dim // 2, 2)
    if debug:
        print("x_reshaped.shape", x_reshaped.shape)
    
    # Create lists to store original and rotated pairs for visualization
    original_pairs = []
    rotated_pairs = []
    
    # Collect original pairs from first head, first position
    for i in range(positions.shape[0]):
        pair = (x_reshaped[0, 0, 0, i, 0].item(), x_reshaped[0, 0, 0, i, 1].item())
        original_pairs.append(pair)

    # 8) Apply RoPE rotation:
    #    Let x_reshaped[..., 0] = x_even
    #        x_reshaped[..., 1] = x_odd
    #    shape: (batch_size, n_heads, seq_len, per_head_dim//2)
    #    Then:
    #       new_even = x_even * cos(theta) - x_odd * sin(theta)
    #       new_odd  = x_even * sin(theta) + x_odd * cos(theta)

    # Extract even and odd indices
    x_even = x_reshaped[..., 0]
    x_odd = x_reshaped[..., 1]
    if debug:
        print("x_even.shape", x_even.shape)
        print("x_odd.shape", x_odd.shape)

    # # Print some example pairs to visualize the structure
    # print("Example even/odd pairs from first head, first position:")
    # for i in range(5):
    #     print(f"Pair {i}: ({x_even[0, 0, i, 0].item():.4f}, {x_odd[0, 0, i, 0].item():.4f})")
    x_rotated_even = x_even * cos_theta - x_odd * sin_theta
    x_rotated_odd  = x_even * sin_theta + x_odd * cos_theta

    # Print some example pairs to visualize the structure
    for i in range(positions.shape[0]):
        pair = (x_rotated_even[0, 0, i, 0].item(), x_rotated_odd[0, 0, i, 0].item())
        rotated_pairs.append(pair)

    if visualize:
        # Calculate number of rows needed (5 items per row)
        num_positions = positions.shape[0]
        num_rows = (num_positions + 4) // 5  # Ceiling division
        items_per_row = 5
        
        # Create a figure with subplots arranged in multiple rows
        # Create the figure and axes
        fig, axes = plt.subplots(num_rows, items_per_row, figsize=(15, 3 * num_rows))
        
        fig.suptitle('RoPE Visualizations: 1 Example, 1 Head, 1 Even/Odd Vector Position Pair', 
            fontsize=16)
        
        # Convert axes to a 2D numpy array regardless of its original shape
        if num_rows == 1 and items_per_row == 1:
            # Single subplot case
            axes = np.array([[axes]])
        elif num_rows == 1:
            # Single row case
            axes = np.array([axes if isinstance(axes, np.ndarray) else [axes]])
        elif items_per_row == 1:
            # Single column case
            axes = np.array([[ax] for ax in axes])
            
        # Colors for original and rotated vectors
        colors = ['blue', 'red']
        labels = ['Original', 'Rotated']

        # Plot each pair in its own subplot
        for i, (orig, rot) in enumerate(zip(original_pairs, rotated_pairs)):
            row, col = i // items_per_row, i % items_per_row
            ax = axes[row, col]
            
            # Plot the original vector
            ax.arrow(0, 0, orig[0], orig[1], head_width=0.05, head_length=0.05, fc=colors[0], ec=colors[0], label=labels[0])
            
            # Plot the rotated vector
            ax.arrow(0, 0, rot[0], rot[1], head_width=0.05, head_length=0.05, fc=colors[1], ec=colors[1], label=labels[1])
            
            # Add a circle to visualize the rotation
            max_radius = max(np.sqrt(orig[0]**2 + orig[1]**2), np.sqrt(rot[0]**2 + rot[1]**2))
            circle = plt.Circle((0, 0), max_radius, fill=False, linestyle='--', alpha=0.3)
            ax.add_patch(circle)
            
            # Set equal aspect ratio and limits
            ax.set_aspect('equal')
            limit = max(max_radius, 0.8) * 1.2
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            
            # Add grid and title
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Sequence Position {past_seq_len + i}')
            
            # Add axes labels
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Add legend (only for the first subplot)
            if i == 0:
                ax.legend()
        
        # Hide unused subplots
        for i in range(num_positions, num_rows * items_per_row):
            row, col = i // items_per_row, i % items_per_row
            if col < axes.shape[1]:  # Check if this column exists
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    # 9) Re-combine the rotated pairs into the last dimension
    #    shape still: (batch_size, seq_len, n_heads, per_head_dim // 2, 2)
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    if debug:
        print("x_rotated.shape", x_rotated.shape)

    # # 10) Reshape x_rotated to match the original shape
    # #    shape: (batch_size, seq_len, n_heads, per_head_dim)
    x_rotated = x_rotated.view(batch_size, n_heads, seq_len, per_head_dim)
    return x_rotated

