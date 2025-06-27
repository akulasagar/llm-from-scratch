import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import os
import math

# Import our GPT model from Phase 9
from phase9_gpt_model import GPTModel

# --- Hyperparameters ---
BATCH_SIZE = 32
CONTEXT_LENGTH = 128
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 200 # How many batches to average for loss estimation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_LAYERS = 6
NUM_HEADS = 6
EMBEDDING_DIM = 384
CHECKPOINT_PATH = 'tinystories_gpt_checkpoint.pth'

# New Learning Rate Scheduler params
LEARNING_RATE = 3e-4 # Max learning rate
MIN_LR = 3e-5        # Min learning rate
WARMUP_ITERS = 200
LR_DECAY_ITERS = MAX_ITERS # Should be equal to MAX_ITERS

# --- 1. Load Data and Tokenizer ---
print("Loading data and tokenizer...")
tokenizer = Tokenizer.from_file('custom_bpe_tokenizer/tokenizer.json')
vocab_size = tokenizer.get_vocab_size()

with open('tinystories_valid.txt', 'r', encoding='utf-8') as f:
    text = f.read()

data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print("Data and tokenizer loaded successfully.")

# --- 2. Data Loading Function ---
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# --- 3. Learning Rate Scheduler Function ---
def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

# --- 4. Model, Optimizer, and Checkpoint Loading ---
model = GPTModel(vocab_size, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, CONTEXT_LENGTH)
m = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start_iter = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter']
    print(f"Resumed from iteration {start_iter}")

print(f"Model loaded on {DEVICE}. Parameter count: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

# --- 5. Loss Estimation ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 6. The Training Loop ---
print("\nStarting advanced training...")
for iter in range(start_iter, MAX_ITERS):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")

        if iter > 0: # Don't save on the first step
            print("Saving checkpoint...")
            checkpoint = { 'iter': iter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': losses['val'] }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print("Checkpoint saved.")

    xb, yb = get_batch('train')
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("\nTraining complete!")

# --- 7. Generate Text from the Model ---
print("\n--- Generating a Tiny Story ---")
# Start with a beginning-of-sequence token if your tokenizer has one, or a common starting word.
# Our BPE tokenizer doesn't have a specific BOS token, so we'll start with a generic prompt.
# Let's encode "Once upon a time"
start_text = "Once upon a time"
start_ids = tokenizer.encode(start_text).ids
context = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

def generate(model, idx, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx

generated_tokens = generate(model, idx=context, max_new_tokens=100)
generated_text = tokenizer.decode(generated_tokens[0].tolist())
print(generated_text)

print("\nFinal model saved to 'tinystories_gpt_model.pth'")
torch.save(model.state_dict(), 'tinystories_gpt_model.pth')