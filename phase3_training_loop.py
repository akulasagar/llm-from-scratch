import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Let's import the code we wrote before.
# We need the dataset from phase 1 and the model from phase 2.
from phase1_mnist_explorer import training_data
from phase2_model_architecture import MNISTClassifier

# --- 1. Hyperparameters and Setup ---
# Hyperparameters are the "settings" of our training process.
LEARNING_RATE = 0.001  # How big of a step the optimizer takes
BATCH_SIZE = 64        # How many images to process at once
EPOCHS = 3             # How many times to loop over the entire dataset

# --- 2. Data Loader ---
# We create a DataLoader to efficiently feed our model batches of data.
train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Model, Loss Function, and Optimizer ---
# Instantiate the model
model = MNISTClassifier()

# Loss Function: CrossEntropyLoss is perfect for multi-class classification.
# It combines the steps of applying Softmax and calculating the negative log-likelihood loss.
loss_fn = nn.CrossEntropyLoss()

# Optimizer: Adam is a great, general-purpose optimizer.
# We tell it which parameters to update (all of them: model.parameters()).
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. The Training Loop ---
print("Starting training...")

# Loop over the dataset for the number of epochs
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    # This will hold the sum of the loss for each batch
    total_loss = 0

    # Loop over each batch from the DataLoader
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Step 1: Forward Pass - get model's predictions (logits)
        logits = model(images)
        
        # Step 2: Calculate Loss - compare predictions to true labels
        loss = loss_fn(logits, labels)
        total_loss += loss.item() # .item() gets the raw number from the tensor

        # Step 3: Backward Pass - calculate gradients
        optimizer.zero_grad() # Reset gradients from previous batch
        loss.backward()       # Compute gradients for this batch

        # Step 4: Optimizer Step - update model weights
        optimizer.step()

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    # Calculate and print average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f"End of Epoch {epoch+1} | Average Training Loss: {avg_loss:.4f}")

print("\nTraining complete!")

# --- 5. Save the Trained Model ---
# The 'state_dict' is a dictionary containing all the learned weights and biases.
# This is the standard way to save a PyTorch model.
torch.save(model.state_dict(), 'mnist_classifier_model.pth')
print("Model weights saved to mnist_classifier_model.pth")