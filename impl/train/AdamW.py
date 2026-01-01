from collections.abc import Callable
import math
from typing import Optional
import torch

class AdamW(torch.optim.Optimizer):
	def __init__(self, params, lr, weight_decay, betas, eps):
		if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
		defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
		super().__init__(params, defaults)

	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()

		for group in self.param_groups:			# For each param_group, they will have different lr
			lr = group["lr"] # Get the learning rate.
			beta1, beta2 = group["betas"]
			eps = group["eps"]
			weight_decay = group["weight_decay"]

			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p] # Get state associated with p.
				# Initialize state if it doesn't exist
				if "m" not in state:
					state["m"] = torch.zeros_like(p.data)
					state["v"] = torch.zeros_like(p.data)
					state["t"] = 0

				m, v = state["m"], state["v"]

				t = state.get("t", 0) + 1 # Get iteration number from the state, or initial value.
				grad = p.grad.data # Get the gradient of loss with respect to p.
				
				# Update biased first and second moment estimates
				m.mul_(beta1).add_(grad, alpha=1 - beta1)
				v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

				lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
				p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)
				p.data = p.data - lr * weight_decay * p.data
				
				# Save state
				state["m"] = m
				state["v"] = v
				state["t"] = t
		return loss
