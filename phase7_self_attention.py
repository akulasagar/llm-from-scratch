import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Setup our inputs ---
# Let's create a small batch of token embeddings to work with.
# Imagine we have a sentence with 5 tokens (sequence_length=5).
# And our embedding dimension is 10 (embedding_dim=10).
# We'll use a batch size of 1 for simplicity.
batch_size = 1
sequence_length = 5
embedding_dim = 10

# This is the shape a real embedding layer would output.
# (Batch Size, Sequence Length, Embedding Dimension)
x = torch.randn(batch_size, sequence_length, embedding_dim)
print(f"Input shape: {x.shape}")


# --- 2. Define linear layers for Q, K, V ---
# In a real model, these would be layers in our nn.Module class.
# For this example, we'll create them as standalone layers.
# Each token's embedding will be passed through these three separate
# linear layers to generate its Query, Key, and Value vectors.

# The output dimension of these layers is the "head dimension".
# It doesn't have to be the same as the embedding dimension.
head_dim = 8

q_layer = nn.Linear(embedding_dim, head_dim, bias=False)
k_layer = nn.Linear(embedding_dim, head_dim, bias=False)
v_layer = nn.Linear(embedding_dim, head_dim, bias=False)

# Generate the Query, Key, and Value matrices
Q = q_layer(x)
K = k_layer(x)
V = v_layer(x)

print(f"\nQ, K, V shapes: {Q.shape}")


# --- 3. Calculate Attention Scores ---
# We calculate scores by taking the dot product of the Query matrix
# with the Key matrix.
# To do this, we need to transpose the Key matrix's last two dimensions.
# Q shape: (1, 5, 8)
# K.transpose(-2, -1) shape: (1, 8, 5)
# Resulting scores shape: (1, 5, 5)
# This (5, 5) matrix shows the attention score of each token with every other token.
scores = torch.matmul(Q, K.transpose(-2, -1))
print(f"\nScores shape (pre-scaling): {scores.shape}")


# --- 4. Scale Scores and Apply Softmax ---
# The dot product scores can get very large, which makes the softmax output
# too "spiky" (one value is almost 1, others are almost 0). This hurts learning.
# We scale them down by the square root of the head dimension.
# This is a trick from the "Attention Is All You Need" paper.
scaled_scores = scores / (head_dim ** 0.5)

# Apply softmax along the last dimension to get attention weights.
# These weights sum to 1 for each row.
attention_weights = F.softmax(scaled_scores, dim=-1)

print(f"\nAttention weights shape: {attention_weights.shape}")
print(f"Example attention weights for the first token:\n{attention_weights[0, 0, :]}")
print(f"Sum of these weights: {attention_weights[0, 0, :].sum()}") # Should be 1.0


# --- 5. Apply Attention Weights to Values ---
# We multiply our attention weights by the Value matrix.
# This "amplifies" the vectors of tokens we want to pay attention to
# and "drowns out" the vectors of irrelevant tokens.
# attention_weights shape: (1, 5, 5)
# V shape: (1, 5, 8)
# Resulting output shape: (1, 5, 8)
output = torch.matmul(attention_weights, V)

print(f"\nFinal output shape: {output.shape}")
print("\nThis output is the new representation of our 5 tokens.")
print("Each token's vector is now a weighted sum of all other token's Value vectors,")
print("enriched with context from the entire sequence.")