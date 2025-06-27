import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Import our GPT model architecture
from phase9_gpt_model import GPTModel

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'tinystories_gpt_model.pth' # The final model from our phase 13 run
TOKENIZER_PATH = 'custom_bpe_tokenizer/tokenizer.json'
CONTEXT_LENGTH = 128

# Generation Parameters
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7  # Higher -> more creative/random, Lower -> more predictable
TOP_K = 50         # Consider only the top 50 most likely tokens

# --- 1. Load Model and Tokenizer ---
print("Loading model and tokenizer...")

# Load tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

# Load model architecture (get hyperparameters from the saved model if possible,
# but for now we'll hardcode them to match our trained model)
model_args = {
    'vocab_size': vocab_size,
    'embedding_dim': 384,
    'num_heads': 6,
    'num_layers': 6,
    'context_length': CONTEXT_LENGTH
}
model = GPTModel(**model_args)

# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval() # Set model to evaluation mode

print("Model and tokenizer loaded successfully.")

# --- 2. The Advanced Generate Function ---
@torch.no_grad()
def generate(start_string, max_new_tokens, temperature, top_k):
    print(f"\n--- Generating with prompt: '{start_string}' ---")
    
    # Encode the starting prompt
    start_ids = tokenizer.encode(start_string).ids
    idx = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        # Crop context if it's too long
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        
        # Forward pass to get logits
        logits = model(idx_cond)
        # Focus on the last token's logits
        logits = logits[:, -1, :] / temperature # (B, C)
        
        # --- Top-k Sampling ---
        # Remove logits for tokens not in the top k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        
        # Sample from the filtered distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # Append the new token and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
    # Decode the generated sequence
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)

# --- 3. Run Generation ---
prompt = "One day, a little rabbit found a"
generated_text = generate(prompt, MAX_NEW_TOKENS, TEMPERATURE, TOP_K)
print("\n--- Generated Story ---")
print(generated_text)

# --- Example of changing parameters ---
print("\n\n--- Now with low temperature (more predictable) ---")
generated_text_low_temp = generate(prompt, MAX_NEW_TOKENS, temperature=0.2, top_k=TOP_K)
print("\n--- Generated Story (Low Temp) ---")
print(generated_text_low_temp)