from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# --- 1. Setup: Create a directory to save the tokenizer ---
TOKENIZER_DIR = 'custom_bpe_tokenizer'
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, 'tokenizer.json')
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# --- 2. Initialize a new BPE Tokenizer ---
# We start with a blank slate.
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# The pre-tokenizer splits the text into words first.
# We'll use a simple whitespace splitter.
tokenizer.pre_tokenizer = Whitespace()

# --- 3. Train the Tokenizer ---
# The trainer will learn the merge rules from our data.
# We need to specify a vocabulary size and special tokens.
# A smaller vocab size is fine for our small dataset.
trainer = BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# List of files to train on
files = ["input.txt"]

# Train the tokenizer
print("Training a new BPE tokenizer...")
tokenizer.train(files, trainer)
print("Training complete!")

# --- 4. Save the Trained Tokenizer ---
tokenizer.save(TOKENIZER_PATH)
print(f"Tokenizer saved to {TOKENIZER_PATH}")

# --- 5. Test the New Tokenizer ---
# Load the tokenizer we just saved
loaded_tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

text = "O Romeo, Romeo! wherefore art thou Romeo?"
print(f"\n--- Testing the new tokenizer ---")
print(f"Original text: '{text}'")

# Encode the text
encoded = loaded_tokenizer.encode(text)

print(f"Encoded IDs: {encoded.ids}")
print(f"Encoded Tokens: {encoded.tokens}")

# Decode the IDs back to text
decoded = loaded_tokenizer.decode(encoded.ids)
print(f"Decoded text: '{decoded}'")

# --- Compare with our old character-level tokenizer ---
char_tokens = list(text)
print("\n--- Comparison ---")
print(f"BPE tokenizer produced {len(encoded.tokens)} tokens.")
print(f"Character tokenizer produced {len(char_tokens)} tokens.")
print("Notice how BPE is much more efficient, grouping common words and punctuation.")