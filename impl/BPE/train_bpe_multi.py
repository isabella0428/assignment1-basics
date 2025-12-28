from multiprocessing import Pool, Process, Manager
from typing import Dict, Tuple
import regex as re
import os
from typing import BinaryIO

# -------------------------
# TokenNode for linked list
# -------------------------
class TokenNode:
    def __init__(self, token: str):
        self.token = token
        self.next = None
        self.prev = None


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
def train_bpe_multi(
	input_path: str,
	vocab_size: int,
	special_tokens: list[str]
)-> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
	# Use a Manager list to share tokens between processes
	manager = Manager()
	tokens = manager.list()

	with open(input_path, "rb") as file:
		num_processes = 4
		boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
	
	# Prepare arguments for pool
	tasks = []
	for i in range(len(boundaries) - 1):
		tasks.append((input_path, boundaries[i], boundaries[i+1], special_tokens))

	with Pool(processes=num_processes) as pool:
		results = pool.starmap(pretokenizer, tasks) 
	
	tokens = [word for sublist in results for word in sublist]
	(vocab, merges) = run_bpe_per_word(tokens, special_tokens, vocab_size)
	return (vocab, merges)

def pretokenizer(input_path: str, start:int, end: int, special_tokens: list[str]):
	with open(input_path, "rb") as file:
		file.seek(start)
		chunk = file.read(end - start).decode("utf-8", errors="ignore")   
	
	# Escape each special token for regex
	escaped_tokens = [re.escape(token) for token in special_tokens]
	# Join them with | to make a regex OR pattern
	pattern = "|".join(escaped_tokens)

	words = re.split(pattern, chunk)

	PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
	final_tokens = []
	for word in words:
		if word in special_tokens:
			continue
		elif word:
			final_tokens.extend(re.findall(PAT, word))
	return final_tokens

def run_bpe_per_word(words: list[str], special_tokens: list[str], vocab_size: int) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
	# Token to the start indexof the token occurrence
	vocabulary = []
	freq_cnt: Dict[Tuple[str, str], int] = {}
	token_pos: Dict[Tuple[str, str], set[TokenNode]] = {}

	# Build initial state
	for word in words:
		nodes = [TokenNode(ch) for ch in word]
		for i in range(len(nodes) - 1):
			nodes[i].next = nodes[i+1]
			nodes[i+1].prev = nodes[i]

			pair = (nodes[i].token, nodes[i+1].token)
			freq_cnt[pair] = freq_cnt.get(pair, 0) + 1
			if pair not in token_pos:
				token_pos[pair] = set()
			token_pos[pair].add(nodes[i])
		vocabulary.append(nodes)

	merges = []
	# 1. Initialize with all 256 bytes + special tokens
	final_vocabulary = {i: bytes([i]) for i in range(256)}
	for i, tok in enumerate(special_tokens):
		final_vocabulary[256 + i] = tok.encode('utf-8')
	
	while len(final_vocabulary) < vocab_size:
		(token1, token2), max_count = max(freq_cnt.items(), key=lambda item: (item[1], item[0]))
		if max_count < 1:
			break

		print(f"Merging token1: {token1} token2: {token2}")
		merges.append((token1.encode('utf-8'), token2.encode('utf-8')))
		final_vocabulary[len(final_vocabulary)] = (token1 + token2).encode('utf-8')

		# Update neighbor counts
		for token_node in list(token_pos[(token1, token2)]):
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
				token_pos[new_prev_pair] = token_pos.get(new_prev_pair, set())
				token_pos[new_prev_pair].add(first.prev)

				# Decrease freq counts
				old_prev_pair = (first.prev.token, first.token)
				freq_cnt[old_prev_pair] = freq_cnt[old_prev_pair] - 1
				token_pos[old_prev_pair].discard(first.prev)
			
			first.next = second_next
			if second_next is not None:
				second_next.prev = first

				# Increment freq counts for the pair including the next node
				next_pair = (new_token_value, second_next.token)
				freq_cnt[next_pair] = freq_cnt.get(next_pair, 0) + 1
				token_pos[next_pair] = token_pos.get(next_pair, set())
				token_pos[next_pair].add(first)

				# Decrease freq counts
				old_prev_pair = (second.token, second_next.token)
				freq_cnt[old_prev_pair] = freq_cnt[old_prev_pair] - 1
				token_pos[old_prev_pair].discard(second)
			
			first.token = new_token_value
			

		del freq_cnt[(token1, token2)]
		del token_pos[(token1, token2)]

	return (final_vocabulary, merges)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
