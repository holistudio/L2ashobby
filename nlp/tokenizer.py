import tiktoken

# Initialize the tokenizer
enc = tiktoken.get_encoding("o200k_base")

# Your token IDs
tokens = [64659, 123310, 75584, 8138, 38271]

# Decode full sequence
decoded_text = enc.decode(tokens)

# Decode each token individually (for inspection)
decoded_tokens = [enc.decode([t]) for t in tokens]

print("Decoded sequence:", decoded_text)
print("Each token separately:", decoded_tokens)
