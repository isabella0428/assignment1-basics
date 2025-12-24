from typing import dict, list, tuple, Iterable, Iterator
import json
from common import gpt2_bytes_to_unicode

class Tokenizer:
	vocab: dict[int, bytes] = {}
	merges: list[tuple[bytes, bytes]] = []
	special_tokens: list[str] | None = None

	def __init__(
		self,
		vocab: dict[int, bytes],
		merges: list[tuple[bytes, bytes]],
		special_tokens: list[str] | None = None):
		self.vocab = vocab
		self.merges = merges
		self.special_tokens = special_tokens

	"""
	Class
	method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
	(in the same format that your BPE training code output) and (optionally) a list of special
	tokens. This method should accept the following additional parameters:
	vocab_filepath: str
	merges_filepath: str
	special_tokens: list[str] | None = None
	"""
	def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
		# Compare the vocab to the expected output vocab
		with open(vocab_filepath, encoding="utf-8") as f:
			vocab_json = json.load(f)
			vocab = {
				vocab_index: bytes([vocab_json[token] for token in vocab_item])
				for vocab_item, vocab_index in vocab_json.items()
			}
		
		# Compare the learned merges to the expected output merges
		gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
		with open(merges_filepath, encoding="utf-8") as f:
			raw_merges = [tuple(line.rstrip().split(" ")) for line in f]
			merges = [
				(
					bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
					bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
				)
				for merge_token_1, merge_token_2 in raw_merges
			]

		cls(vocab, merges, special_tokens)



	"""
	Encode an input text into a sequence of token IDs.
	"""
	def encode(self, text: str) -> list[int]:
		pass
	
	"""
	 Given an iterable of
	strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
	required for memory-eﬀicient tokenization of large files that we cannot directly load into
	memory.
	"""
	def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
		pass

	"""
	Decode a sequence of token IDs into text.
	"""
	def decode(self, ids: list[int]) -> str:
		pass
		