# phase1_mnist_explorer.py 

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_mnist_data():
    """Downloads the MNIST dataset and returns the training and test sets."""
    print("Downloading MNIST data...")
    
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download training data
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    # Download test data
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    
    print("Download complete!")
    return train_data, test_data

# This block of code will only run when you execute this script directly
# e.g., `python phase1_mnist_explorer.py`
# It will NOT run when this file is imported by another script.
if __name__ == '__main__':
    # Get the data
    training_data, test_data = get_mnist_data()

    print(f"Number of training images: {len(training_data)}")
    print(f"Number of test images: {len(test_data)}")

    # Create a temporary DataLoader to explore the first image
    temp_loader = torch.utils.data.DataLoader(training_data, batch_size=1)
    first_image, first_label = next(iter(temp_loader))

    print("\nExploring the first image:")
    print(f"The label is: {first_label.item()}")
    print(f"The image is a tensor of shape: {first_image.shape}")

    print("\nDisplaying the first image...")
    image_to_show = first_image.squeeze()
    plt.imshow(image_to_show, cmap='gray')
    plt.title(f"Label: {first_label.item()}")
    plt.show()