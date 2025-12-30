import torch
from einops import einsum
from impl.transformer.Softmax import Softmax
import numpy as np

class ScaledDotProductAttention(torch.nn.Module):
	def apply(
			self,
			q: torch.Tensor,		# (batch_size, ..., seq_len, d_k)
			k: torch.Tensor,		# (batch_size, ..., seq_len, d_k)
			v: torch.Tensor,		# (batch_size, ..., seq_len, d_v)
			mask: torch.Tensor		# (seq_len, seq_len)
	) -> torch.Tensor:	# (batch_size, ..., d_v)
		d_k = q.size(-1)
		sqrt = np.sqrt(d_k)
		pre_softmax = einsum(
			q,
			k,
			"... seq_len_1 d_k, ... seq_len_2 d_k -> ... seq_len_1 seq_len_2"
		) / sqrt

		if mask is not None:
            # We use a very large negative number for numerical stability
			pre_softmax = pre_softmax.masked_fill(mask == False, float("-inf"))

		after_softmax = Softmax().apply(pre_softmax, -1)
		attention = after_softmax @ v
		return attention