import torch

class Embedding(torch.nn.Module):
	"""
	Construct an embedding module. This function should accept the following parameters:
	num_embeddings: int Size of the vocabulary
	embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
	device: torch.device | None = None Device to store the parameters on
	dtype: torch.dtype | None = None Data type of the parameters
	"""
	def __init__(self, num_embeddings: int, embedding_dim: int, device:torch.device | None = None, dtype: torch.dtype | None = None):
		super().__init__()

		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.device = device
		self.dtype = dtype

		self.embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
		
		# N(µ = 0, σ2 =1 truncated at [−3σ, 3σ].
		sigma = 1
		self.embedding_matrix = torch.nn.init.trunc_normal_(self.embedding_matrix, mean=0, std = sigma, a = -3 * sigma, b = 3 * sigma)

		self.embedding_matrix = torch.nn.Parameter(self.embedding_matrix)
	
	"""
	Lookup the embedding vectors for the given token IDs
	token_ids: with size (batch_size, sequence_length)
	"""
	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		return self.embedding_matrix[token_ids]
