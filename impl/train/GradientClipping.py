import torch
from typing import Iterable

class GradientClipping():
	@staticmethod
	def apply(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
		total_norm_sq = 0.0
		grads = [p.grad for p in parameters if p.grad is not None]
		
		for grad in grads:
			grad_norm = torch.norm(grad, p=2)
			total_norm_sq += grad_norm ** 2

		total_norm = total_norm_sq ** 0.5
		if total_norm > max_l2_norm:
			factor = max_l2_norm / (total_norm + eps)
			for grad in grads:
				grad.data.mul_(factor)
