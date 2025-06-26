import pickle

# --- 1. Load the dataset ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("--- Dataset Info ---")
print(f"Length of dataset in characters: {len(text):,}")

# --- 2. Create the Vocabulary ---
# Find all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {''.join(chars)}")
print(f"Vocabulary size: {vocab_size}")

# --- 3. Create the Tokenizer ---
# We create a mapping from characters to integers (and vice-versa)
stoi = { ch:i for i,ch in enumerate(chars) } # string-to-integer
itos = { i:ch for i,ch in enumerate(chars) } # integer-to-string

# The encoder takes a string and outputs a list of integers
def encode(s):
    return [stoi[c] for c in s]

# The decoder takes a list of integers and outputs a string
def decode(l):
    return ''.join([itos[i] for i in l])

# --- 4. Test the Tokenizer ---
print("\n--- Testing the Tokenizer ---")
test_string = "Hello, world!"
encoded_output = encode(test_string)
decoded_output = decode(encoded_output)

print(f"Original string: '{test_string}'")
print(f"Encoded output (token IDs): {encoded_output}")
print(f"Decoded back to string: '{decoded_output}'")
assert test_string == decoded_output
print("Test successful!")

# --- 5. Save the tokenizer metadata ---
# We need to save the `stoi`, `itos`, and `vocab_size` so our training
# script can use them. We'll use pickle to save these Python objects.
tokenizer_data = {
    'stoi': stoi,
    'itos': itos,
    'vocab_size': vocab_size,
}
with open('tokenizer_meta.pkl', 'wb') as f:
    pickle.dump(tokenizer_data, f)
    
print("\nTokenizer metadata saved to 'tokenizer_meta.pkl'")