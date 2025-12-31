from typing import Dict
import torch

from impl.transformer.RMSNorm import RMSNorm
from impl.transformer.SwiGLU import SwiGLU
from impl.transformer.MultiHeadSelfAttentionWithRope import MultiHeadSelfAttentionWithRope

class TransformerLayer(torch.nn.Module):
	"""
	Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your
	Transformer block should accept (at least) the following parameters.
	
	d_model (int): The dimensionality of the Transformer block input.
	num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
		evenly divisible by `num_heads`.
	d_ff (int): Dimensionality of the feed-forward inner layer.
	max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
	theta (float): RoPE parameter.
	weights (dict[str, Tensor]):
		State dict of our reference implementation.
		The keys of this dictionary are:
		- `attn.q_proj.weight`
			The query projections for all `num_heads` attention heads.
			Shape is (d_model, d_model).
			The rows are ordered by matrices of shape (num_heads, d_k),
			so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
		- `attn.k_proj.weight`
			The key projections for all `num_heads` attention heads.
			Shape is (d_model, d_model).
			The rows are ordered by matrices of shape (num_heads, d_k),
			so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
		- `attn.v_proj.weight`
			The value projections for all `num_heads` attention heads.
			Shape is (d_model, d_model).
			The rows are ordered by matrices of shape (num_heads, d_v),
			so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
		- `attn.output_proj.weight`
			Weight of the multi-head self-attention output projection
			Shape is (d_model, d_model).
		- `ln1.weight`
			Weights of affine transform for the first RMSNorm
			applied in the transformer block.
			Shape is (d_model,).
		- `ffn.w1.weight`
			Weight of the first linear transformation in the FFN.
			Shape is (d_model, d_ff).
		- `ffn.w2.weight`
			Weight of the second linear transformation in the FFN.
			Shape is (d_ff, d_model).
		- `ffn.w3.weight`
			Weight of the third linear transformation in the FFN.
			Shape is (d_model, d_ff).
		- `ln2.weight`
			Weights of affine transform for the second RMSNorm
			applied in the transformer block.
			Shape is (d_model,).
	in_features (Float[Tensor, "batch sequence_length d_model"]):
		Tensor to run your implementation on.
	"""
	def apply(
			self,
		   	d_model: int,
			num_heads: int,
			d_ff: int,
			max_seq_len: int,
			theta: float,
			weights: Dict[str, torch.Tensor],
			in_features: torch.Tensor) -> torch.Tensor:
		# Attention layer rmsNorm
		rmsNorm1 = RMSNorm(d_model)
		rmsNorm1.load_state_dict({"g": weights["ln1.weight"]})
		attention_layer_rms_x = rmsNorm1.forward(in_features)
		
		# Multi head self attention
		seq_len = in_features.shape[-2]
		token_positions = torch.arange(seq_len, device=in_features.device)
		attention_result = MultiHeadSelfAttentionWithRope().apply(
			num_heads,
			theta,
			max_seq_len,
			weights["attn.q_proj.weight"],
			weights["attn.k_proj.weight"],
			weights["attn.v_proj.weight"],
			weights["attn.output_proj.weight"],
			attention_layer_rms_x,
			token_positions
		)
		attention_layer_result = in_features + attention_result

		# FFN layer rmsNorm
		rmsNorm2 = RMSNorm(d_model)
		rmsNorm2.load_state_dict({"g": weights["ln2.weight"]})
		rmsNorm2_result = rmsNorm2.forward(attention_layer_result)

		# SwiGLU
		swiGLU = SwiGLU(d_model, d_ff)
		swiGLU.load_state_dict({"W1": weights["ffn.w1.weight"], "W2": weights["ffn.w2.weight"], "W3": weights["ffn.w3.weight"]})
		swiGLU_result = swiGLU.forward(rmsNorm2_result)
		return attention_layer_result + swiGLU_result
