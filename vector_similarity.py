# Step 1: Import the libraries
import numpy as np
import torch

# Step 2: Define word vectors using NumPy
# Let's pretend these vectors represent the "meaning" of the words "cat" and "kitten"
vector_cat = np.array([0.8, 0.2, 0.9, 0.1])
vector_kitten = np.array([0.7, 0.3, 0.8, 0.2])

print("--- Using NumPy ---")
print(f"Vector for 'cat': {vector_cat}")
print(f"Vector for 'kitten': {vector_kitten}")


# Step 3: Calculate the Euclidean Distance with NumPy
# The formula is: sqrt(sum((A - B)^2))
difference = vector_cat - vector_kitten
squared_diff = difference ** 2
sum_squared_diff = np.sum(squared_diff)
numpy_distance = np.sqrt(sum_squared_diff)

print(f"Euclidean Distance (NumPy): {numpy_distance}\n")


# Step 4: Convert to PyTorch Tensors
print("--- Using PyTorch ---")
tensor_cat = torch.from_numpy(vector_cat)
tensor_kitten = torch.from_numpy(vector_kitten)
# We can also create them directly: tensor_cat = torch.tensor([0.8, 0.2, 0.9, 0.1])

print(f"Tensor for 'cat': {tensor_cat}")
print(f"Tensor for 'kitten': {tensor_kitten}")


# Step 5: Calculate Distance with PyTorch
# PyTorch has a built-in function to make this easy
# torch.cdist needs the tensors to be 2D, so we add an extra dimension with .unsqueeze(0)
pytorch_distance = torch.cdist(tensor_cat.unsqueeze(0), tensor_kitten.unsqueeze(0))

print(f"Euclidean Distance (PyTorch): {pytorch_distance.item()}")