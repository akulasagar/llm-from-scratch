import torch
import torch.nn as nn

# --- 1. Setup our Embedding Layer ---

# Let's imagine a very small vocabulary, like the one for our tokenizer.
# We'll use the GPT-2 vocab size for this example.
vocab_size = 50257

# This is the size of our "meaning" vector. Every token will be represented
# by a list of this many numbers. This is a key hyperparameter in any LLM.
# For GPT-2 small, this is 768. Let's use a smaller, more viewable number.
embedding_dim = 10

# Create the embedding layer. It's essentially a big lookup table.
# It will have `vocab_size` rows and `embedding_dim` columns.
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

print(f"--- Embedding Layer Info ---")
print(f"Embedding layer created for a vocabulary of {vocab_size} tokens.")
print(f"Each token will be represented by a vector of size {embedding_dim}.")
print(f"Shape of the embedding weight matrix: {embedding_layer.weight.shape}")

# --- 2. Get Embeddings for our Token IDs ---

# Let's use the token IDs from our previous script.
# text = "Hello, world! Let's build our own Large Language Model."
input_ids = torch.tensor([15496, 11, 995, 0, 3914, 338, 1382, 674, 898, 13601, 15417, 9104, 13])

# The embedding layer expects a tensor of IDs as input.
# The input tensor must be of type LongTensor (64-bit integers).
input_ids = input_ids.long()

print(f"\n--- Looking up embeddings ---")
print(f"Shape of our input IDs tensor: {input_ids.shape}")

# Pass the IDs through the embedding layer
# This is the core operation: looking up the vector for each ID.
word_embeddings = embedding_layer(input_ids)

print(f"Shape of the output embeddings tensor: {word_embeddings.shape}")
print("\nThis shape means we have 13 tokens, and each one is now a 10-dimensional vector.")

# --- 3. Let's inspect one embedding ---
print("\n--- Inspecting a single embedding ---")
hello_id = 15496
hello_embedding = embedding_layer(torch.tensor(hello_id).long())

world_id = 995
world_embedding = embedding_layer(torch.tensor(world_id).long())

print(f"The embedding for 'Hello' (ID {hello_id}):\n{hello_embedding}")
print(f"\nThe embedding for 'world' (ID {world_id}):\n{world_embedding}")
print("\nNote: These vectors are currently random. The goal of training is to make them meaningful.")