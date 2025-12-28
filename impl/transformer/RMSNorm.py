import torch

class RMSNorm(torch.nn.Module):
	"""
	Construct the RMSNorm module. This function should accept the following parameters:
	d_model: int Hidden dimension of the model
	eps: float = 1e-5 Epsilon value for numerical stability
	device: torch.device | None = None Device to store the parameters on
	dtype: torch.dtype | None = None Data type of the parameters
	"""
	def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
		super().__init__()
		
		self.d_model = d_model
		self.eps = eps
		self.device = device
		self.dtype = dtype

		# Gain parameter
		self.g = torch.nn.Parameter(torch.randn(d_model, device=device, dtype=dtype))

	"""
	Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
	"""
	def forward(self, x: torch.Tensor) -> torch.Tensor: 
		in_dtype = x.dtype
		x = x.to(torch.float32)
		# Square, Mean over last dim, Add eps, Sqrt
        # keepdim=True changes shape from (B, L) to (B, L, 1) for broadcasting
		ms = x.pow(2).mean(dim=-1, keepdim=True)
		rms = torch.sqrt(ms + self.eps)

		# x is (B, L, D)
		# self.g is (D, ) => stretches to (B, L, D)
		# rms is (B, L, 1)

		# The broadcasting rule check dim from right to left:
		# 1. if one dimension is 1 and the other is D, stretch to D
		# 2. if match proceeds
		# 3. if not match, throws an error
		return (x * self.g / rms).to(dtype=in_dtype)