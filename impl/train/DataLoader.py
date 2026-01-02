import numpy as np
import numpy.typing as npt
import torch

class DataLoader():
	@staticmethod
	def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str)-> tuple[torch.Tensor, torch.Tensor]:
		length = dataset.shape[-1]

		batch_sample_data = []
		batch_label_data = []
		for _ in range(batch_size):
			start_index = np.random.randint(0, high=length - context_length, size=None, dtype=int)
			end_index = start_index + context_length - 1
			sample_data = dataset[start_index : end_index+1]
			batch_sample_data.append(sample_data)

			label_data = dataset[start_index+1: end_index+2]
			batch_label_data.append(label_data)

		return (torch.tensor(batch_sample_data), torch.tensor(batch_label_data))

