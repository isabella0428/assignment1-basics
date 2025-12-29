import torch

class SwiGLU(torch.nn.Module):
	def __init__(self, d_model: int, d_ff: int):
		super().__init__()

		self.W1 = torch.nn.Parameter(
			torch.randn(
				d_ff,
				d_model
			)
		)

		self.W2 = torch.nn.Parameter(
			torch.randn(
				d_model,
				d_ff
			)
		)

		self.W3 = torch.nn.Parameter(
			torch.randn(
				d_ff,
				d_model
			)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 1. Project to intermediate dimension (d_ff)
        # x is (..., d_model), W1 is (d_ff, d_model)
		w1X = x @ self.W1.T 
		w3X = x @ self.W3.T

		siLu_w1x = w1X * torch.sigmoid(w1X) # (..., d_ff)
		swiGLU = (siLu_w1x * w3X) @ self.W2.T # (..., d_model)
		return swiGLU