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
        print("positions.shape", positions.shape, positions)

    # 3) Create an index over the half-dimension. We treat each dimension i as paired (2i, 2i+1).
    #    shape: (per_head_dim//2,)
    dim_idx = torch.arange(per_head_dim // 2)
    if debug:
        print("dim_idx.shape", dim_idx.shape, dim_idx)

    # 4) Compute the "angle" or "theta" for each pair of dimensions:
    #    RoPE defines these angles as positions * (base_freq ^ (-2*i / d))
    #    where d is the total head dimension. We exponentiate freq in the negative direction.
    #    shape: (seq_len, per_head_dim//2)
    theta = positions.unsqueeze(1) * (freq ** (-2 * dim_idx / per_head_dim))
    if debug:
        print("theta.shape", theta.shape)
        print("theta", theta)

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
        print("cos_theta", cos_theta)
        print("sin_theta.shape", sin_theta.shape)
        print("sin_theta", sin_theta)

    # 7) Reshape x so we can work with the final dimension as pairs: (..., 2)
    #    shape: (batch_size, n_heads, seq_len, per_head_dim//2, 2)
    x_reshaped = x.view(batch_size, n_heads, seq_len, per_head_dim // 2, 2)
    if debug:
        print("x_reshaped.shape", x_reshaped.shape)
        print("x_reshaped", x_reshaped[0, 0, :1])
        # print("x_reshaped", x_reshaped[0, 0, 2:4])
    
    # Create lists to store original and rotated pairs for visualization
    original_pairs = []
    rotated_pairs = []
    
    # Collect original pairs from first head, first position
    for i in range(per_head_dim // 2):
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
        print("x_even", x_even[0, 0, :1])
        print("x_odd.shape", x_odd.shape)
        print("x_odd", x_odd[0, 0, :1])

    # # Print some example pairs to visualize the structure
    # print("Example even/odd pairs from first head, first position:")
    # for i in range(5):
    #     print(f"Pair {i}: ({x_even[0, 0, i, 0].item():.4f}, {x_odd[0, 0, i, 0].item():.4f})")
    x_rotated_even = x_even * cos_theta - x_odd * sin_theta
    x_rotated_odd  = x_even * sin_theta + x_odd * cos_theta

    # Print detailed calculations for up to 2 sequence positions
    for pos in range(min(2, seq_len)):
        print(f"\nMath equation for position {pos}:")
        
        # First dimension calculations
        first_even = x_even[0, 0, pos, 0].item()
        first_odd = x_odd[0, 0, pos, 0].item()
        first_cos = cos_theta[0, 0, pos, 0].item()
        first_sin = sin_theta[0, 0, pos, 0].item()
        
        print(f"Dimension 0:")
        print(f"x_rotated_even = {first_even:.4f} * {first_cos:.4f} - {first_odd:.4f} * {first_sin:.4f} = {first_even * first_cos - first_odd * first_sin:.4f}")
        print(f"x_rotated_odd = {first_even:.4f} * {first_sin:.4f} + {first_odd:.4f} * {first_cos:.4f} = {first_even * first_sin + first_odd * first_cos:.4f}")
        
        # Second dimension calculations
        second_even = x_even[0, 0, pos, 1].item()
        second_odd = x_odd[0, 0, pos, 1].item()
        second_cos = cos_theta[0, 0, pos, 1].item()
        second_sin = sin_theta[0, 0, pos, 1].item()
        
        print(f"Dimension 1:")
        print(f"x_rotated_even = {second_even:.4f} * {second_cos:.4f} - {second_odd:.4f} * {second_sin:.4f} = {second_even * second_cos - second_odd * second_sin:.4f}")
        print(f"x_rotated_odd = {second_even:.4f} * {second_sin:.4f} + {second_odd:.4f} * {second_cos:.4f} = {second_even * second_sin + second_odd * second_cos:.4f}")
    
    # Print the actual computed values
    print("\nComputed rotated values:")
    for pos in range(min(2, seq_len)):
        print(f"Position {pos}:")
        print(f"x_rotated_even: {x_rotated_even[0, 0, pos, :2]}")
        print(f"x_rotated_odd: {x_rotated_odd[0, 0, pos, :2]}")

    # Collect rotated pairs
    for i in range(per_head_dim // 2):
        pair = (x_rotated_even[0, 0, 0, i].item(), x_rotated_odd[0, 0, 0, i].item())
        rotated_pairs.append(pair)
        print("rotated_pairs", len(rotated_pairs))

    if visualize:
        # Colors for original and rotated vectors
        colors = ['blue', 'red']
        labels = ['Original', 'Rotated']

        # Visualize dimensions for up to 2 sequence positions
        for pos in range(min(2, seq_len)):
            print(f"\nPosition {pos} dimensions:")
            
            # Print dimension information
            for dim in range(per_head_dim // 2):
                print(f"Dimension {dim}: Original=({x_even[0, 0, pos, dim].item():.4f}, {x_odd[0, 0, pos, dim].item():.4f}), "
                      f"Rotated=({x_rotated_even[0, 0, pos, dim].item():.4f}, {x_rotated_odd[0, 0, pos, dim].item():.4f})")
            
            # Calculate grid dimensions
            num_dims = per_head_dim // 2
            cols = min(5, num_dims)  # Maximum 5 columns
            rows = (num_dims + cols - 1) // cols  # Ceiling division
            
            # Create figure with appropriate grid
            pos_fig, pos_axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
            pos_fig.suptitle(f'Position {pos} - All Dimensions', fontsize=14)
            
            # Convert axes to 2D array if it's not already
            if rows == 1 and cols == 1:
                pos_axes = np.array([[pos_axes]])
            elif rows == 1:
                pos_axes = pos_axes.reshape(1, -1)
            elif cols == 1:
                pos_axes = pos_axes.reshape(-1, 1)
            
            # Plot each dimension pair
            for dim in range(num_dims):
                row = dim // cols
                col = dim % cols
                ax = pos_axes[row, col]
                
                # Get original and rotated vectors for this dimension
                orig = (x_even[0, 0, pos, dim].item(), x_odd[0, 0, pos, dim].item())
                rot = (x_rotated_even[0, 0, pos, dim].item(), x_rotated_odd[0, 0, pos, dim].item())
                
                # Plot the original vector
                ax.arrow(0, 0, orig[0], orig[1], head_width=0.05, head_length=0.05, 
                         fc=colors[0], ec=colors[0], label=labels[0])
                
                # Plot the rotated vector
                ax.arrow(0, 0, rot[0], rot[1], head_width=0.05, head_length=0.05, 
                         fc=colors[1], ec=colors[1], label=labels[1])
                
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
                ax.set_title(f'Dimension Pair k={dim}')
                
                # Add axes labels
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                
                # Add legend (only for the first subplot)
                if dim == 0:
                    ax.legend()
            
            # Hide unused subplots
            for dim in range(num_dims, rows * cols):
                row = dim // cols
                col = dim % cols
                pos_axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()

    # 9) Re-combine the rotated pairs into the last dimension
    #    shape still: (batch_size, seq_len, n_heads, per_head_dim // 2, 2)
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    if debug:
        print("x_rotated.shape", x_rotated.shape)
        # print("x_rotated", x_rotated[0, 0, :1])

    # # 10) Reshape x_rotated to match the original shape
    # #    shape: (batch_size, seq_len, n_heads, per_head_dim)
    x_rotated = x_rotated.view(batch_size, n_heads, seq_len, per_head_dim)
    if debug:
        print("x_rotated", x_rotated[0, 0, :1])
    return x_rotated

