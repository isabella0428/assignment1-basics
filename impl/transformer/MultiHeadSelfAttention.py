import torch
from impl.transformer.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadSelfAttention(torch.nn.Module):
	"""
	num_heads (int): Number of heads to use in multi-headed attention.
	q_proj_weight (Float[Tensor, "h*d_k d_in"]): Weights for the Q projection
	k_proj_weight (Float[Tensor, "h*d_k d_in"]): Weights for the K projection
	v_proj_weight (Float[Tensor, "h*d_k d_in"]): Weights for the V projection
	o_proj_weight (Float[Tensor, "d_model h*d_v"]): Weights for the output projection
	in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
	"""
	def apply(
		self,
		num_heads: int,
		q_proj_weight: torch.Tensor,
		k_proj_weight: torch.Tensor,
		v_proj_weight: torch.Tensor,
		o_proj_weight: torch.Tensor,
		in_features: torch.Tensor
	):
		*batch_dims, seq_len, _ = in_features.shape
		d_k = q_proj_weight.shape[-2] // num_heads
		d_v = v_proj_weight.shape[-2] // num_heads

		q = in_features @ q_proj_weight.T		# (..., seq_len, h*d_k)
		k = in_features @ k_proj_weight.T		# (..., seq_len, h*d_k)
		v = in_features @ v_proj_weight.T		# (..., seq_len, h*d_v)

		q = q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)	# (*batch_dims, num_heads, seq_len, d_k)
		k = k.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)  # (*batch_dims, num_heads, seq_len, d_k)
		v = v.view(*batch_dims, seq_len, num_heads, d_v).transpose(-3, -2)  # (*batch_dims, num_heads, seq_len, d_v)

		mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
		mask = mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
		mask = mask.expand(*batch_dims, num_heads, seq_len, seq_len)  # shape (*batch_dims, num_heads, seq_len, seq_len)
		mask = mask.to(q.device)

		attn_out = ScaledDotProductAttention().apply(q, k, v, mask)
		attn_out = attn_out.transpose(-3, -2).contiguous()
		concat_out = attn_out.view(*batch_dims, seq_len, num_heads * d_v)
		return torch.matmul(concat_out, o_proj_weight.transpose(-1, -2))
