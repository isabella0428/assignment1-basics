import torch
import numpy as np

class RotaryPositionalEmbedding(torch.nn.Module):
	"""
	Construct he RoPE module and create buffers if needed.
	theta: float Θ value for the RoPE
	d_k: int dimension of query and key vectors
	max_seq_len: int Maximum sequence length that will be inputted
	device: torch.device | None = None Device to store the buffer on
	"""
	def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
		super().__init__()
		self.theta = theta
		rotation_matrix = torch.zeros(max_seq_len, d_k, d_k, device=device)

		half_d = d_k // 2
		for pos in range(max_seq_len):
			k = torch.arange(half_d, device=device)
			angles = pos / (theta ** (2 * k / d_k))
			cos = torch.cos(angles)
			sin = torch.sin(angles)

            # Fill 2x2 blocks
			for i in range(half_d):
				rotation_matrix[pos, 2*i:2*i+2, 2*i:2*i+2] = torch.tensor([
					[cos[i], -sin[i]],
   					[sin[i],  cos[i]]
				], device=device, dtype=rotation_matrix.dtype)

		self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)


	"""
	Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
	Note that you should tolerate x with an arbitrary number of batch dimensions. You should
	assume that the token positions are a tensor of shape (..., seq_len) specifying the token
	positions of x along the sequence dimension.
	You should use the token positions to slice your (possibly precomputed) cos and sin tensors
	along the sequence dimension.
	"""
	def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
		*batch_dims, seq_len, d_k = x.shape
		x_flat = x.reshape(-1, seq_len, d_k)
		token_pos_flat = token_positions.reshape(-1, seq_len)  # (batch_flat, seq_len)

        # Select rotation matrices for each token position
		rot = self.rotation_matrix[token_pos_flat]  # (batch_flat, seq_len, d_k, d_k)

        # Batch matrix multiplication: R @ x
		x_flat_unsq = x_flat.unsqueeze(-1)  # (batch_flat, seq_len, d_k, 1)
		out_flat = torch.matmul(rot, x_flat_unsq)  # (batch_flat, seq_len, d_k, 1)
		out_flat = out_flat.squeeze(-1)           # (batch_flat, seq_len, d_k)
		return out_flat.reshape(*batch_dims, seq_len, d_k)