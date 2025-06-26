import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# Import our functions and classes
from phase1_mnist_explorer import get_mnist_data
from phase2_model_architecture import MNISTClassifier

# --- 1. Load the Test Data ---
_, test_data = get_mnist_data()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# --- 2. Load the Trained Model ---
# First, create an instance of the model architecture
model = MNISTClassifier()
# Then, load the saved weights (the state_dict) into the model instance
model.load_state_dict(torch.load('mnist_classifier_model.pth'))
# Set the model to evaluation mode. This is important for layers like Dropout or BatchNorm.
model.eval()

# --- 3. Evaluate the Model on the Test Set ---
correct_predictions = 0
total_samples = 0

# `with torch.no_grad():` tells PyTorch we don't need to calculate gradients here,
# which saves memory and computation.
with torch.no_grad():
    for images, labels in test_loader:
        # Get model predictions
        logits = model(images)
        
        # The output logits are raw scores. We need to find the index of the
        # highest score, which corresponds to the predicted class (digit).
        # torch.max returns both the max value and its index. We only want the index.
        _, predicted = torch.max(logits, dim=1)
        
        # Update counts
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = (correct_predictions / total_samples) * 100
print(f"Model Accuracy on the test set: {accuracy:.2f}%")

# --- 4. Inference on a Single Random Image ---
print("\n--- Running inference on a single test image ---")
# Get a random image from the test dataset
random_idx = random.randint(0, len(test_data) - 1)
single_image, single_label = test_data[random_idx]

# The model expects a batch, so we add a "batch" dimension of size 1
# Original shape: [1, 28, 28] -> New shape: [1, 1, 28, 28]
image_for_model = single_image.unsqueeze(0)

with torch.no_grad():
    logits = model(image_for_model)
    _, predicted_label = torch.max(logits, dim=1)

# Display the image and the prediction
plt.imshow(single_image.squeeze(), cmap='gray')
plt.title(f"True Label: {single_label} | Predicted: {predicted_label.item()}")
plt.show()