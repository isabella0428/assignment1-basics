from typing import List


def pretokenizer(text: str)-> List[str]:
	return text.split(" ")

def BPE_within_word(vocabularies, rounds) -> List[str]:
	for r in range(rounds):
		freq = {}
		max_pair_start_idx = -1
		max_pair = ""
		for i in range(1, len(vocabularies)):
			pair = vocabularies[i-1] + vocabularies[i]
			freq[pair] = freq.get(pair, 0) + 1
			if len(max_pair) == 0 or freq[max_pair] < freq[pair] or (freq[max_pair] == freq[pair] and max_pair < pair):
				max_pair = pair
				max_pair_start_idx = i-1

		updated_vocabularies = []
		i = 0
		while i < len(vocabularies)-1:
			if vocabularies[i] == vocabularies[max_pair_start_idx] and vocabularies[i+1] == vocabularies[max_pair_start_idx+1]:
				updated_vocabularies.append(max_pair)
				i += 1
			else:
				updated_vocabularies.append(vocabularies[i])
				if i == len(vocabularies)-2:
					updated_vocabularies.append(vocabularies[i+1])
			i = i + 1

		print(f"Round {r}, merge result {vocabularies[max_pair_start_idx]} {vocabularies[max_pair_start_idx + 1]}")
		vocabularies = updated_vocabularies
		print(f'updated vocabularies: {updated_vocabularies}')


if __name__ == "__main__":
	text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
	# Tokenize by whitespace
	tokenizedStrs = pretokenizer(text)
	print("Tokens:", tokenizedStrs)

	# Flatten tokens into a list of characters
	vocabularies = [c for token in tokenizedStrs for c in token]
	print("Initial vocabularies:", vocabularies)
	BPE_within_word(vocabularies, 6)
