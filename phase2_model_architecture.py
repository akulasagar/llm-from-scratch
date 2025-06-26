import torch
import torch.nn as nn

# We will use a simple Feed-Forward Neural Network, also called a
# Multi-Layer Perceptron (MLP).

class MNISTClassifier(nn.Module):
    def __init__(self):
        # This is the constructor. It runs once when we create a model instance.
        # We define the layers of our network here.
        super().__init__() # This line is mandatory boilerplate.

        # Our images are 28x28 pixels. To feed them into a simple network,
        # we need to "flatten" them from a 2D grid into a 1D list.
        # 28 * 28 = 784.
        # So, our input layer will have 784 features.
        input_size = 784

        # This is the number of neurons in our hidden layer. It's a "hyperparameter"
        # that we can tune. 128 is a reasonable starting point.
        hidden_size = 128

        # The output of our model needs to be 10 numbers, one for each digit (0-9).
        # These are often called "logits".
        output_size = 10

        # Define the sequence of layers
        self.network = nn.Sequential(
            # Layer 1: Flattens the 28x28 image into a 784-element vector.
            nn.Flatten(),

            # Layer 2: A linear layer that takes 784 inputs and maps them to 128 outputs.
            nn.Linear(input_size, hidden_size),

            # Layer 3: An activation function. ReLU is a standard, effective choice.
            # It introduces non-linearity, which allows the model to learn complex patterns.
            nn.ReLU(),

            # Layer 4: The final linear layer. It takes the 128 features from the
            # hidden layer and maps them to our 10 output classes.
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # This method defines the "forward pass". It tells the model how data
        # flows through the layers we defined in __init__.
        # 'x' will be our batch of input images.
        return self.network(x)


# --- Let's test our model definition ---
if __name__ == '__main__':
    # Create an instance of our model
    model = MNISTClassifier()
    print("Model architecture:")
    print(model)

    # Let's create a fake image tensor to see what happens when we pass it through.
    # The shape is [1, 1, 28, 28] which means:
    # (1 batch item, 1 color channel, 28 pixels high, 28 pixels wide)
    fake_image = torch.randn(1, 1, 28, 28)

    # Pass the fake image through the model
    output = model(fake_image)

    print("\n--- Testing the forward pass ---")
    print(f"Shape of the fake image input: {fake_image.shape}")
    print(f"Shape of the model's output: {output.shape}")
    print(f"Raw output (logits) for the fake image:\n{output}")