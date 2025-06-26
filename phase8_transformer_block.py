import torch
import torch.nn as nn

# We are re-implementing the Self-Attention Head as a proper nn.Module
# to make it reusable and part of the larger computation graph.
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embedding_dim, head_dim, bias=False)
        self.k = nn.Linear(embedding_dim, head_dim, bias=False)
        self.v = nn.Linear(embedding_dim, head_dim, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, embedding_dim)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        # output shape: (batch, seq_len, head_dim)
        return output

# Now we build Multi-Head Attention by combining several heads
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        
        # Create a list of attention heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, self.head_dim) for _ in range(num_heads)
        ])
        
        # A final linear layer to project the concatenated heads back to embedding_dim
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # Run each head in parallel
        outputs = [head(x) for head in self.heads]
        # Concatenate the head outputs along the last dimension
        concatenated = torch.cat(outputs, dim=-1)
        # Pass through the final output layer
        projected_output = self.output_layer(concatenated)
        return projected_output

# A simple Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), # Expand
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)  # Project back
        )
    
    def forward(self, x):
        return self.net(x)

# Finally, we assemble the full Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForward(embedding_dim)
        # Layer Normalization helps stabilize the network during training
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # This is the "Pre-LN" variant used in many modern models like GPT-2
        
        # First sub-layer: Multi-Head Attention
        # The "x +" part is the residual connection. It helps prevent gradients from vanishing.
        x = x + self.attention(self.ln1(x))
        
        # Second sub-layer: Feed-Forward Network
        x = x + self.ffn(self.ln2(x))
        
        return x

# --- Let's test our Transformer Block ---
if __name__ == '__main__':
    embedding_dim = 12
    num_heads = 4 # So each head will have a dimension of 12 / 4 = 3
    sequence_length = 5
    batch_size = 1

    # Create a Transformer Block
    block = TransformerBlock(embedding_dim, num_heads)
    
    # Create a fake input tensor
    x = torch.randn(batch_size, sequence_length, embedding_dim)
    
    print("--- Testing the Transformer Block ---")
    print(f"Input shape: {x.shape}")
    
    # Pass the input through the block
    output = block(x)
    
    print(f"Output shape: {output.shape}")
    print("\nThe output shape is the same as the input shape, which is crucial.")
    print("The block has processed the information, but maintained the dimensionality.")
    print("An LLM is just a stack of these blocks.")