import torch
import math

class Linear(torch.nn.Module):
	"""
	Construct a
	linear transformation module. This function should accept the following parameters:
	in_features: int final dimension of the input
	out_features: int final dimension of the output
	device: torch.device | None = None Device to store the parameters on
	dtype: torch.dtype | None = None Data type of the parameters
	"""
	def __init__(self, in_features: int, out_features: int, device: torch.device|None=None, dtype: torch.dtype=None):
		super().__init__()

		self.W = torch.empty(out_features, in_features, device=device, dtype=dtype)

		# N(µ = 0, σ2 =2 / (din +dout) truncated at [−3σ, 3σ].
		sigma = math.sqrt(2.0 / (in_features + out_features))
		self.W = torch.nn.init.trunc_normal_(self.W, mean=0, std = sigma, a = -3 * sigma, b = 3 * sigma)

		# Convert to parameter
		self.W = torch.nn.Parameter(self.W)

	#  Apply the linear transformaation to the input
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x @ self.W.T
