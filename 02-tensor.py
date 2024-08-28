# Tensors are a specialized data structure that are very similar to arrays and matrices
# In Pytorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters

# Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or other hardware accelerators
# In fact, tensors and Numpy arrays can often share the same underlying memory, eliminating the need to copy data
# Tensors are also optimized for automatic differenciation

import torch
import numpy as np

## Initializing a Tensor
# Tensors can be initialized in various ways

# Directly from data: tensors can be created directly from data. teh data type is automatically inferred
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a Numpy array
# Tensors can be created from Numpy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Fro manother tensor: The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor:\n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor:\n{x_rand}\n")

# With random or constant values: shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor
shape = (3, 4,) # (number of row, number of column)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}\n")

# Attributes of a Tensor: Tensor attributes describe their shape, datatype, and the device on which they are stored
tensor = torch.rand(3, 4)
print(f"shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on {tensor.device}")

print(tensor)

## Operations on Tensors
# Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively contained
# Each of these operations can be run on the GPU
# By default, tensors are created on the CPU
# We need to explicitly move tensors to the GPU using .to method (after checking for GPU availability)
# Keep in mind that copying large tensors across devices can be expensive in terms of time and memory

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

tensor = tensor.to(device)

# Standard numpy-like indexing and sclicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:,1] = 0

print(tensor, "\n")

# Joining tensors: can use torch.cat to concatenate a sequence of tensors along a given dimension
t1 = torch.cat([tensor, tensor, tensor], dim = 0)
t2 = torch.cat([tensor, tensor, tensor], dim = 1)

print(t1, "\n")
print(t2, "\n")

## Arithmetic Operations
# Matrix multiplication
# y1, y2, y3 will have the same value
# tensor.T returns tensor's transpose
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)

# element-wise product
# z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)

print(f"y1: {y1}\n")
print(f"y2: {y2}\n")
print(f"y3: {y3}\n")

print(f"z1: {z1}\n")
print(f"z2: {z2}\n")
print(f"z3: {z3}\n")

## Single-element tensors: If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item()
agg = tensor.sum()
agg_item = agg.item()
print(f"agg value: {agg}\n agg type: {type(agg)}")
print(f"agg_item value: {agg_item}\nagg item type: {type(agg_item)}")

## In-place operations: Operations that store the result into the operand are called in-place. They are denoted by a _ suffix
# For example: x.copy_(y), x.t_(), will change x
# In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history
# Hence, their use is discouraged
print(f"{tensor}\n")
tensor.add_(5)
print(tensor)

## Bridge with Numpy
# Tensors on teh CPU and Numpy arrays can share their underlying memory locations, and changing one will change the other

# tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the Numpy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the numpy array reflects in the tensor
np.add(n, 1, out = n)
print(f"t: {t}")
print(f"n: {n}")
