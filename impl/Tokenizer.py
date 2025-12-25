from typing import Dict, List, Tuple, Iterable, Iterator
import json
from tests import common
import heapq
import regex as re

# -------------------------
# TokenNode for linked List
# -------------------------
class TokenNode:
	def __init__(self, token: str):
		self.token = token
		self.next = None
		self.prev = None
	
	def __lt__(self, other):
		return self.token < other.token

class Tokenizer:
	vocab: Dict[int, bytes] = {}
	merges: List[Tuple[bytes, bytes]] = []
	special_tokens: List[str] | None = None

	def __init__(
		self,
		vocab: Dict[int, bytes],
		merges: List[Tuple[bytes, bytes]],
		special_tokens: List[str] | None = None):
		self.vocab = vocab
		self.merges = merges
		self.special_tokens = special_tokens
		# Regex for pre-tokenization (Matches the training PAT)
		self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
		if self.special_tokens is None:
			self.special_tokens = []

	"""
	Class method that constructs and return a Tokenizer from a serialized vocabulary and List of merges
	(in the same format that your BPE training code output) and (optionally) a List of special
	tokens. This method should accept the following additional parameters:
	vocab_filepath: str
	merges_filepath: str
	special_tokens: List[str] | None = None
	"""
	def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
		# Compare the vocab to the expected output vocab
		with open(vocab_filepath, encoding="utf-8") as f:
			vocab_json = json.load(f)
			vocab = {
				vocab_index: [vocab_json[token] for token in vocab_item]
				for vocab_item, vocab_index in vocab_json.items()
			}
		
		# Compare the learned merges to the expected output merges
		gpt2_byte_decoder = {v: k for k, v in common.gpt2_bytes_to_unicode().items()}
		with open(merges_filepath, encoding="utf-8") as f:
			raw_merges = [tuple(line.rstrip().split(" ")) for line in f]
			merges = [
				(
					bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
					bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
				)
				for merge_token_1, merge_token_2 in raw_merges
			]

		return cls(vocab, merges, special_tokens)

	"""
	Encode an input text into a sequence of token IDs.
	"""
	def encode(self, text: str) -> List[int]:
		if not text:
			return []
		
		token_to_id = {v: k for k, v in self.vocab.items()}

		# Create a pattern: (<|endoftext|>|<|other|>)
		parts = [text]
		if len(self.special_tokens) > 0:
			special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
			special_pat = "(" + "|".join(re.escape(t) for t in special_tokens_sorted) + ")"
			parts = re.split(special_pat, text)
		
		# Split the text
		# "A<|endoftext|>B" -> ["A", "<|endoftext|>", "B"]
		final_ids = []
		for part in parts:
			if part in self.special_tokens:
				token_bytes = part.encode('utf-8')
				final_ids.append(token_to_id[token_bytes])
			else:
				pretokens = self.pat.findall(part)
				for chunk in pretokens:
					chunk_bytes = chunk.encode('utf-8')
					final_ids.extend(self._encode_chunk(chunk_bytes))
				
		return final_ids
	
	def _encode_chunk(self, text: bytes) -> List[int]:
		nodes = [TokenNode(bytes([b])) for b in text]

		for i in range(len(nodes) - 1):
			nodes[i].next = nodes[i + 1]
			nodes[i + 1].prev = nodes[i]

		# ---------- helpers ----------
		merge_rank = {pair: i for i, pair in enumerate(self.merges)}
		token_to_id = {v: k for k, v in self.vocab.items()}

		 # ---------- initialize heap ----------
		heap = []
		for i in range(len(nodes) - 1):
			pair = (bytes(nodes[i].token), bytes(nodes[i + 1].token))
			if pair in merge_rank:
				heapq.heappush(heap, (merge_rank[pair], (nodes[i], nodes[i + 1])))

		while heap:
			(rank, (left, right)) = heapq.heappop(heap)

			# skip stale entries
			if left.next is not right or right.prev is not left:
				continue

			# Both nodes might be updated with their neighbors
			pair = (left.token, right.token)
			if pair not in merge_rank or merge_rank[pair] != rank:
				continue

			left.token = left.token + right.token
			left.next = right.next

			 # check new left pair
			if left.prev:
				left.prev.next = left
				prev_pair = (bytes(left.prev.token), bytes(left.token))
				if prev_pair in merge_rank:
					heapq.heappush(
						heap,
						(merge_rank[prev_pair], (left.prev, left))
					)

			# check new right pair
			if left.next:
				left.next.prev = left
				next_pair = (bytes(left.token), bytes(left.next.token))
				if next_pair in merge_rank:
					heapq.heappush(
						heap,
						(merge_rank[next_pair], (left, left.next))
					)

		# ---------- collect result ----------
		result = []
		cur = nodes[0]
		while cur:
			result.append(token_to_id[bytes(cur.token)])
			cur = cur.next
		return result

	"""
	Given an iterable of
	strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
	required for memory-eﬀicient tokenization of large files that we cannot directly load into
	memory.
	"""
	def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
		for chunk in iterable:
			# encode each chunk independently
			for token_id in self.encode(chunk):
				yield token_id

	"""
	Decode a sequence of token IDs into text.
	"""
	def decode(self, ids: List[int]) -> str:
		# Join bytes then decode once for safety with multi-byte characters
		byte_sequence = b"".join([self.vocab[idx] for idx in ids])
		return byte_sequence.decode('utf-8', errors='replace')