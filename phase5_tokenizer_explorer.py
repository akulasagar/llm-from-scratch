from transformers import AutoTokenizer

# This line downloads the tokenizer used by the GPT-2 model.
# We are using a pre-trained tokenizer for now to understand the concept.
# Later, we will train our own from scratch.
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("--- Vocabulary Info ---")
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
# The pad_token is used to make all sequences in a batch the same length.
print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
# The EOS token marks the end of a sequence.
print(f"End of Sequence (EOS) token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

# --- Let's tokenize some text ---
text = "Hello, world! Let's build our own Large Language Model."
print(f"\nOriginal text:\n'{text}'")

# The main function of the tokenizer is to encode text into a list of numbers (token IDs).
encoded_ids = tokenizer.encode(text)

print(f"\nEncoded Token IDs:\n{encoded_ids}")
print(f"Number of tokens: {len(encoded_ids)}")

# We can also see what the actual tokens are. Notice how "Language" and "Model" are
# kept as single tokens, but "tokenizer" would be split.
decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
print(f"\nDecoded Tokens:\n{decoded_tokens}")

# --- Now, let's go back from IDs to text ---
# The decode function reconstructs the string from the token IDs.
reconstructed_text = tokenizer.decode(encoded_ids)
print(f"\nReconstructed text:\n'{reconstructed_text}'")

# --- Let's try a more complex example with sub-words ---
complex_text = "Tokenization is an art. Unhappiness is decomposable."
print(f"\n--- Complex Example ---")
print(f"Original complex text:\n'{complex_text}'")

complex_ids = tokenizer.encode(complex_text)
complex_tokens = tokenizer.convert_ids_to_tokens(complex_ids)
print(f"\nComplex text tokens:\n{complex_tokens}")