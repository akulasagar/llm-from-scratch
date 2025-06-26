import torch
import torch.nn as nn
from phase8_transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, context_length):
        """
        Args:
            vocab_size (int): The number of unique tokens in our vocabulary.
            embedding_dim (int): The dimensionality of the token embeddings.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of Transformer blocks to stack.
            context_length (int): The maximum sequence length the model can handle.
        """
        super().__init__()
        self.context_length = context_length
        
        # 1. Token and Positional Embedding
        # The token embedding table, same as before
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        # The positional embedding table. This learns a unique vector for each position.
        self.position_embedding_table = nn.Embedding(context_length, embedding_dim)
        
        # 2. The stack of Transformer Blocks
        # nn.Sequential allows us to chain layers together.
        # We use a list comprehension to create `num_layers` of our block.
        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        
        # 3. Final Layer Normalization
        # A final LayerNorm after the blocks, as is common.
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        # 4. The Language Model Head
        # This is the final linear layer that maps the output of the Transformer
        # back to the vocabulary size, giving us logits for the next token.
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        # idx shape: (batch, sequence_length)
        batch, seq_len = idx.shape
        
        # Get token embeddings
        tok_emb = self.token_embedding_table(idx) # (batch, seq_len, embedding_dim)
        
        # Get positional embeddings
        # torch.arange creates a tensor like [0, 1, 2, ..., seq_len-1]
        pos_emb = self.position_embedding_table(torch.arange(seq_len)) # (seq_len, embedding_dim)
        
        # Add them together. Broadcasting handles the batch dimension.
        x = tok_emb + pos_emb # (batch, seq_len, embedding_dim)
        
        # Pass through the stack of Transformer blocks
        x = self.blocks(x)
        
        # Pass through the final layer norm
        x = self.ln_f(x)
        
        # Pass through the language model head to get logits
        logits = self.lm_head(x) # (batch, seq_len, vocab_size)
        
        return logits


# --- Let's test our complete GPT Model ---
if __name__ == '__main__':
    # Define hyperparameters for a "toy" GPT model
    vocab_size = 50257 # From our GPT-2 tokenizer
    embedding_dim = 96  # Must be divisible by num_heads
    num_heads = 6
    num_layers = 4
    context_length = 128 # The model can look at up to 128 tokens in the past
    
    # Create the model
    model = GPTModel(vocab_size, embedding_dim, num_heads, num_layers, context_length)
    print("--- GPT Model Architecture ---")
    print(model)
    
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Create a fake input batch of token IDs
    # (batch_size=2, sequence_length=10)
    idx = torch.randint(0, vocab_size, (2, 10))
    
    print("\n--- Testing the forward pass ---")
    print(f"Input shape (token IDs): {idx.shape}")
    
    # Get the model's output (logits)
    logits = model(idx)
    
    print(f"Output shape (logits): {logits.shape}")
    print("\nThe output shape is (batch, seq_len, vocab_size), as expected.")
    print("For each of the 10 tokens in the sequence, the model has produced a logit")
    print("for every possible next token in the vocabulary.")