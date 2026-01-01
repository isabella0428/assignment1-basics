import torch
from jaxtyping import Float, Int

class CrossEntrophy(torch.nn.Module):
	"""Given a tensor of inputs and targets, compute the average cross-entropy
	loss across examples.

	Args:
		inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
			unnormalized logit of jth class for the ith example.
		targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
			Each value must be between 0 and `num_classes - 1`.

	Returns:
		Float[Tensor, ""]: The average cross-entropy loss across examples.
	"""
	@staticmethod
	def apply(
		inputs: Float[torch.Tensor, " batch_size vocab_size"],
		targets: Int[torch.Tensor, " batch_size"]):

		batch_size = inputs.shape[-2]

		# Subtract the largest element for numerical stability
		max_val = torch.amax(inputs, dim=-1, keepdim=True)
		updated_inputs = inputs - max_val

		# Get corresonding logits based on target index
		target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

		# Re-adding the max_val of inputs back
		p = torch.log(torch.sum(torch.exp(updated_inputs), dim=-1)) + max_val.squeeze(-1) - target_logits

		return torch.sum(p, dim=-1) / batch_size


