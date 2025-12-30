import torch
from einops import einsum
from Softmax import Softmax

class ScaledDotProductAttention(torch.nn.Module):
	def apply(
			q: torch.Tensor,		# (batch_size, ..., seq_len, d_k)
			k: torch.Tensor,		# (batch_size, ..., seq_len, d_k)
			v: torch.Tensor,		# (batch_size, ..., seq_len, d_v)
			mask: torch.Tensor		# (seq_len, seq_len)
	) -> torch.Tensor:	# (batch_size, ..., d_v)
		d_k = q.shape(-1)
		sqrt = torch.sqrt(d_k)
		pre_softmax = einsum(
			q,
			k,
			"... seq_len_1 d_k, ... seq_len_2 d_k -> ... seq_len_1 seq_len_2"
		) / sqrt

		if mask is not None:
            # We use a very large negative number for numerical stability
			pre_softmax = pre_softmax.masked_fill(mask == False, float("-inf"))

		after_softmax = Softmax().apply(pre_softmax, dim=-1)
		attention = after_softmax @ v
		return attention