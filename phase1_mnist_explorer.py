import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- 1. Define a transform to normalize the data ---
# PyTorch's datasets return images in a certain format (PILImage).
# We need to convert them to Tensors so our model can work with them.
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalizes the tensor with a mean and standard deviation.
    # For MNIST, the mean is 0.1307 and std is 0.3081. This helps training.
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- 2. Download the training and test datasets ---
# The torchvision library makes this incredibly easy.
# It will download the data to a 'data' folder in your project if it doesn't exist.
print("Downloading MNIST training data...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

print("Downloading MNIST test data...")
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("\nDownload complete!")
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of test images: {len(test_dataset)}")


# --- 3. Explore a single data point ---
# Let's grab the very first image and its label from the training set.
# A dataset in PyTorch is like a list, you can access items with [index].
image, label = train_dataset[0]

print(f"\nExploring the first image:")
print(f"The label is: {label}")
# The image is a tensor. .shape tells us its dimensions.
# [1, 28, 28] means: 1 color channel (it's grayscale), 28 pixels high, 28 pixels wide.
print(f"The image is a tensor of shape: {image.shape}")


# --- 4. Visualize the image ---
# We use matplotlib to actually see the image.
# We need to remove the color channel dimension for plotting, so we use .squeeze().
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show() # This will pop up a window with the image.