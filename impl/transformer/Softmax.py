import torch

class Softmax(torch.nn.Module):
	def apply(self, x: torch.Tensor, dimension: int) -> torch.Tensor:
		max_values, _ = torch.max(x, dim=dimension, keepdim=True)
		x = x - max_values
		exp_sum = torch.exp(x).sum(dim=dimension, keepdim=True)
		exp = torch.exp(x)
		return exp / exp_sum
