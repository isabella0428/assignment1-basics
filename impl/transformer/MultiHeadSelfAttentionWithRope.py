import torch
from impl.transformer.ScaledDotProductAttention import ScaledDotProductAttention
from impl.transformer.RotaryPositionalEmbedding import RotaryPositionalEmbedding

class MultiHeadSelfAttentionWithRope(torch.nn.Module):
	"""
	Args:
        num_heads (int): Number of heads to use in multi-headed attention.
        theta (float): RoPE parameter.
		max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
	"""
	def apply(
		self,
		num_heads: int,
		theta: float,
		max_seq_len: int,
		q_proj_weight: torch.Tensor,
		k_proj_weight: torch.Tensor,
		v_proj_weight: torch.Tensor,
		o_proj_weight: torch.Tensor,
		in_features: torch.Tensor,
		token_positions: torch.Tensor
	):
		*batch_dims, seq_len, _ = in_features.shape
		d_k = q_proj_weight.shape[-2] // num_heads
		d_v = v_proj_weight.shape[-2] // num_heads

		q = in_features @ q_proj_weight.T		# (..., seq_len, h*d_k)
		k = in_features @ k_proj_weight.T		# (..., seq_len, h*d_k)
		v = in_features @ v_proj_weight.T		# (..., seq_len, h*d_v)

		rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
		q = q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)	# (*batch_dims, num_heads, seq_len, d_k)
		k = k.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)  # (*batch_dims, num_heads, seq_len, d_k)
		v = v.view(*batch_dims, seq_len, num_heads, d_v).transpose(-3, -2)  # (*batch_dims, num_heads, seq_len, d_v)

		q_after_rope = rope.forward(q, token_positions)
		k_after_rope = rope.forward(k, token_positions)

		mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
		mask = mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
		mask = mask.expand(*batch_dims, num_heads, seq_len, seq_len)  # shape (*batch_dims, num_heads, seq_len, seq_len)
		mask = mask.to(q.device)

		attn_out = ScaledDotProductAttention().apply(q_after_rope, k_after_rope, v, mask)
		attn_out = attn_out.transpose(-3, -2).contiguous()
		concat_out = attn_out.view(*batch_dims, seq_len, num_heads * d_v)
		return torch.matmul(concat_out, o_proj_weight.transpose(-1, -2))
