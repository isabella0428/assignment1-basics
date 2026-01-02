import numpy as np
from typing import Dict, List
from impl.BPE.Tokenizer import Tokenizer
from impl.BPE.train_bpe_multi import train_bpe_multi

from impl.transformer.TransformerLM import TransformerLM

from impl.train.DataLoader import DataLoader
from impl.train.Checkpoint import Checkpoint
from impl.train.AdamW import AdamW
from impl.train.CrossEntrophy import CrossEntrophy

def run_bpe(file_path: str, vocab_size: int, special_tokens: List[str] | None = None):
	return train_bpe_multi(file_path, vocab_size, special_tokens=special_tokens)
	
def tokenize(text: str, vocab: Dict[int, bytes], merges: List[tuple[bytes]]):
	tokenizer = Tokenizer(vocab, merges)
	return tokenizer.encode(text)

if __name__ == "__main__":
	dataset_path = "data/owt_valid.txt"
	batch_size = 20
	vocab_size = 50000
	special_tokens = []
	context_length = 100
	d_model = 400
	d_ff = 800
	num_layers = 5
	num_heads = 2
	rope_theta = 0.1
	transformer_weights = {}
	iterations = 100
	# Optimizers
	lr = 0.01
	weight_decay = 0.01
	betas = (0.01, 0.02)
	eps = 0.01

	with open(dataset_path, 'r', encoding='utf-8') as f:
		raw_text = f.read()
		(vocab, merges) = run_bpe(dataset_path, vocab_size, special_tokens)
		tokenized = tokenize(raw_text, vocab, merges)
		np.save("data/owt_valid_tokenized.npy", tokenized)
		dataset = np.load("data/owt_valid_tokenized.npy", mmap_mode='r')

		transformerLM = TransformerLM()
		optimizer = AdamW(transformerLM.parameters(), lr, weight_decay, betas, eps)
		
		for i in range(iterations):
			(X, Y) = DataLoader.get_batch(dataset, batch_size, context_length)
			
			logits = transformerLM.apply(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, transformer_weights, X)

			# Compute loss (cross-entropy)
			loss = CrossEntrophy(
				logits.view(-1, logits.size(-1)),
				Y.view(-1)
			)

			# Backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Save state dict for next iteration
			transformer_weights = transformerLM.state_dict()
			if (i % 10 == 0 and i > 0):
				checkpoint_file = f"checkpoint-{i}"
				Checkpoint.save(transformerLM, optimizer, i)
