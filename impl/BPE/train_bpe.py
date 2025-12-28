from typing import Dict, List, Tuple
import regex as re

"""
input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
lary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.
"""
def train_bpe(
	input_path: str,
	vocab_size: int,
	special_tokens: list[str]
)-> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
	with open(input_path, "r") as file:
		content = file.read()
		token_list_without_special_chars = remove_special_chars(content, special_tokens)
		token_after_pretokenizer = pretokenizer(token_list_without_special_chars)
		(vocab, merges) = run_bpe_per_word(token_after_pretokenizer, special_tokens, vocab_size)
	return (vocab, merges)

def remove_special_chars(content: str, special_tokens: list[str]) -> list[str]:
    # Escape each special token for regex
    escaped_tokens = [re.escape(token) for token in special_tokens]
    # Join them with | to make a regex OR pattern
    pattern = "|".join(escaped_tokens)
    # Split content by the pattern
    return re.split(pattern, content)

def pretokenizer(words: list[str]) -> list[str]:
	PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
	token_lists = [re.findall(PAT, word) for word in words]
	flattened = [token for sublist in token_lists for token in sublist]
	return flattened

def run_bpe_per_word(words: list[str], special_tokens: list[str], vocab_size: int) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
	# Token to the start indexof the token occurrence
	vocabulary = []
	freq_cnt: Dict[Tuple[str, str], int] = {}
	token_pos: Dict[Tuple[str, str], list[TokenNode]] = {}

	# Build initial state
	for word in words:
		nodes = [TokenNode(ch) for ch in word]
		for i in range(len(nodes) - 1):
			nodes[i].next = nodes[i+1]
			nodes[i+1].prev = nodes[i]

			pair = (nodes[i].token, nodes[i+1].token)
			freq_cnt[pair] = freq_cnt.get(pair, 0) + 1
			token_pos[pair] = token_pos.get(pair, []) + [nodes[i]]
		vocabulary.append(nodes)

	merges = []
	# 1. Initialize with all 256 bytes + special tokens
	final_vocabulary = {i: bytes([i]) for i in range(256)}
	for i, tok in enumerate(special_tokens):
		final_vocabulary[256 + i] = tok.encode('utf-8')
	
	while len(final_vocabulary) < vocab_size:
		(token1, token2), max_count = max(freq_cnt.items(), key=lambda item: (item[1], item[0]))
		if max_count == 1:
			break

		print(f"Merging token1: {token1} token2: {token2}")
		merges.append((token1.encode('utf-8'), token2.encode('utf-8')))
		final_vocabulary[len(final_vocabulary)] = (token1 + token2).encode('utf-8')

		# Update neighbor counts
		for token_node in token_pos[(token1, token2)]:
			first = token_node
			second = first.next
			if second is None:
				continue  # Safety check

			# Merge the tokens
			new_token_value =  first.token + second.token
			second_next = second.next

			if first.prev is not None:
				# Increment freq counts for the pair including the previous node
				new_prev_pair = (first.prev.token, new_token_value)
				freq_cnt[new_prev_pair] = freq_cnt.get(new_prev_pair, 0) + 1
				token_pos[new_prev_pair] = token_pos.get(new_prev_pair, []) + [first.prev]

				# Decrease freq counts
				old_prev_pair = (first.prev.token, first.token)
				freq_cnt[old_prev_pair] = freq_cnt[old_prev_pair] - 1
				token_pos[old_prev_pair].remove(first.prev)

			
			first.next = second_next
			if second_next is not None:
				second_next.prev = first

				# Increment freq counts for the pair including the next node
				next_pair = (new_token_value, second_next.token)
				freq_cnt[next_pair] = freq_cnt.get(next_pair, 0) + 1
				token_pos[next_pair] = token_pos.get(next_pair, []) + [first]

				# Decrease freq counts
				old_prev_pair = (second.token, second_next.token)
				freq_cnt[old_prev_pair] = freq_cnt[old_prev_pair] - 1
				token_pos[old_prev_pair].remove(second)
			
			first.token = new_token_value
			

		del freq_cnt[(token1, token2)]
		del token_pos[(token1, token2)]

	return (final_vocabulary, merges)

class TokenNode:
    def __init__(self, token):
        self.token = token
        self.next = None
        self.prev = None

