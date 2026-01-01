from typing import Dict
import torch

from impl.transformer.Embedding import Embedding
from impl.transformer.TransformerBlock import TransformerBlock
from impl.transformer.RMSNorm import RMSNorm
from impl.transformer.Linear import Linear

class TransformerLM(torch.nn.Module):
	"""
	vocab_size (int): The number of unique items in the output vocabulary to be predicted.
	context_length (int): The maximum number of tokens to process at once.
	d_model (int): The dimensionality of the model embeddings and sublayer outputs.
	num_layers (int): The number of Transformer layers to use.
	num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
		evenly divisible by `num_heads`.
	d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
	rope_theta (float): The RoPE $\Theta$ parameter.
	weights (dict[str, Tensor]):
		State dict of our reference implementation. {num_layers} refers to an
		integer between `0` and `num_layers - 1` (the layer index).
		The keys of this dictionary are:
		- `token_embeddings.weight`
			Token embedding matrix. Shape is (vocab_size, d_model).
		- `layers.{num_layers}.attn.q_proj.weight`
			The query projections for all `num_heads` attention heads.
			Shape is (num_heads * (d_model / num_heads), d_model).
			The rows are ordered by matrices of shape (num_heads, d_k),
			so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
		- `layers.{num_layers}.attn.k_proj.weight`
			The key projections for all `num_heads` attention heads.
			Shape is (num_heads * (d_model / num_heads), d_model).
			The rows are ordered by matrices of shape (num_heads, d_k),
			so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
		- `layers.{num_layers}.attn.v_proj.weight`
			The value projections for all `num_heads` attention heads.
			Shape is (num_heads * (d_model / num_heads), d_model).
			The rows are ordered by matrices of shape (num_heads, d_v),
			so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
		- `layers.{num_layers}.attn.output_proj.weight`
			Weight of the multi-head self-attention output projection
			Shape is ((d_model / num_heads) * num_heads, d_model).
		- `layers.{num_layers}.ln1.weight`
			Weights of affine transform for the first RMSNorm
			applied in the transformer block.
			Shape is (d_model,).
		- `layers.{num_layers}.ffn.w1.weight`
			Weight of the first linear transformation in the FFN.
			Shape is (d_model, d_ff).
		- `layers.{num_layers}.ffn.w2.weight`
			Weight of the second linear transformation in the FFN.
			Shape is (d_ff, d_model).
		- `layers.{num_layers}.ffn.w3.weight`
			Weight of the third linear transformation in the FFN.
			Shape is (d_model, d_ff).
		- `layers.{num_layers}.ln2.weight`
			Weights of affine transform for the second RMSNorm
			applied in the transformer block.
			Shape is (d_model,).
		- `ln_final.weight`
			Weights of affine transform for RMSNorm applied to the output of the final transformer block.
			Shape is (d_model, ).
		- `lm_head.weight`
			Weights of the language model output embedding.
			Shape is (vocab_size, d_model).
	in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
		`sequence_length` is at most `context_length`.
	"""
	def apply(
			self,
		   	vocab_size: int,
		   	context_length: int,
			d_model: int,
			num_layers: int,
			num_heads: int,
			d_ff: int,
			rope_theta:float,
			weights: Dict[str, torch.Tensor],
			in_indices: torch.Tensor) -> torch.Tensor:
		# Embedding
		embedding = Embedding(vocab_size, d_model)
		embedding.load_state_dict({"embedding_matrix": weights["token_embeddings.weight"]})
		embedded_matrix = embedding.forward(
			in_indices
		)

		# Transformer blocks
		transformer_layer_output = embedded_matrix
		for i in range(num_layers):
			block_weights = {
				"attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
				"attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
				"attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
				"attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
				"ln1.weight": weights[f"layers.{i}.ln1.weight"],
				"ln2.weight": weights[f"layers.{i}.ln2.weight"],
				"ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
				"ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
				"ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
			}
			transformer_layer_output = TransformerBlock().apply(
				d_model, num_heads, d_ff, context_length, rope_theta, block_weights, transformer_layer_output
			)

		# RMS norm
		final_rmsNorm = RMSNorm(d_model)
		final_rmsNorm.load_state_dict({"g": weights["ln_final.weight"]})
		final_rmsNorm_result = final_rmsNorm.forward(transformer_layer_output)

		# final linear
		linear = Linear(d_model, vocab_size)
		linear.load_state_dict({"W": weights["lm_head.weight"]})
		final_linear_result = linear.forward(final_rmsNorm_result)
		return final_linear_result
