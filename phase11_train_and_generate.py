import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os  # To check for file existence

# Import our GPT model from Phase 9
from phase9_gpt_model import GPTModel

# --- Hyperparameters ---
BATCH_SIZE = 64
CONTEXT_LENGTH = 128
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_LAYERS = 6
NUM_HEADS = 6
EMBEDDING_DIM = 384
CHECKPOINT_PATH = 'shakespeare_gpt_checkpoint.pth'

# --- 1. Load Data and Tokenizer ---
print("Loading data and tokenizer...")
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('tokenizer_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']

# Encoder/Decoder from our tokenizer script
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the entire dataset to a single tensor of token IDs
data = torch.tensor(encode(text), dtype=torch.long)

# Split into training and validation sets (90% train, 10% val)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- 2. Data Loading Function ---
def get_batch(split):
    # Select train or validation data
    data = train_data if split == 'train' else val_data
    # Generate random starting points for our batches
    ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCH_SIZE,))
    # Create input sequences (x)
    x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
    # Create target sequences (y), which are shifted by one position
    y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
    # Move data to the selected device (GPU or CPU)
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# --- 3. Model, Optimizer, and Checkpoint Loading ---
model = GPTModel(vocab_size, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, CONTEXT_LENGTH)
m = model.to(DEVICE)  # Move the model to the GPU
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Load from checkpoint if it exists
start_iter = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter']
    print(f"Resumed from iteration {start_iter}")

print(f"Model loaded on {DEVICE}. Parameter count: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

# A function to estimate the loss without calculating gradients
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_INTERVAL)
        for k in range(EVAL_INTERVAL):
            X, Y = get_batch(split)
            logits = model(X)
            # Reshape logits and targets for cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = F.cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set model back to training mode
    return out

# --- 4. The Training Loop ---
print("\nStarting training...")
for iter in range(start_iter, MAX_ITERS):
    # Every once in a while, evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save checkpoint
        print("Saving checkpoint...")
        checkpoint = {
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses['val'],
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print("Checkpoint saved.")

    # Get a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    yb = yb.view(B*T)
    loss = F.cross_entropy(logits, yb)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete!")

# --- 5. Generate Text from the Model ---
print("\n--- Generating Text ---")
# Start with a single "newline" character as the initial prompt
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

def generate(model, idx, max_new_tokens):
    model.eval() # Set model to evaluation mode for generation
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop idx to the last CONTEXT_LENGTH tokens
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        # Get the predictions
        logits = model(idx_cond)
        # Focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    model.train() # Set model back to training mode
    return idx

generated_tokens = generate(model, idx=context, max_new_tokens=500)
generated_text = decode(generated_tokens[0].tolist())
print(generated_text)

# Save the final trained model
torch.save(model.state_dict(), 'shakespeare_gpt_model.pth')
print("\nFinal model saved to 'shakespeare_gpt_model.pth'")