# This section runs through the APT for common tasks in machine learning

## Working with data
# Pytorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset
# Dataset stores the samples and their corresponding labels
# DataLoader wraps an iterable around the Dataset
# The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO
# In this tutorial, we use the fashionMNIST dataset
# Every TorchVision Dataset includes two arguments: transform and target_transform to modify the samples and labels respectively
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Downlad training data from open datasets
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

# Download test data from open datasets
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

# We pass the Dataset as an argument to DataLoader
# This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading

# Defining a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels
batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader:
    # N: batch size
    # C: Number of Channel(1 = black and white image, 3 = RGB color image)
    # H: Height
    # W: Width
    print(f"shape of X [N, C, H, W]: {X.shape}")
    print(f"shape of y: {y.shape} {y.dtype}\n")
    break

## Creating Models
# To define a neural network in Pytorch, we create a class that inherits from nn.Module
# We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function
# To accelerate operation in the neural network, we move it to the GPU or MPS if available

# Get cpu, gpu or mps device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"using {device} device\n")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model, "\n")


## Optimizing the Model Parameters
# To train a model, we need a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

# In a single training loop, the model makes predictions on the training dataset (fet to it in batches), and backpropagates the prediction error to adjust the model's parameters
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# We also check the model's performance against the test dataset to ensure it is learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"\nTest Error\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

# The training process is conducted over several iterations (epochs)
# During each epoch, the model learns parameters to make better predictions
# We print the model's accuracy and loss at each epoch; we'd loke to see the accuracy increase and the loss decreasewith every epoch 
epochs = 10
for t in range(epochs):
    print(f"Epoch: {t+1}\n---------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!\n")


## Saving models
# Common way to save a model is to serialize the internal state dictionary (containing the model parameters)


# torch.save(model.state_dict(), "model.pth")
# print("Saved Pytorch Model State to model.pth")

## Loading Models
# The process for loading a model includes re-creating the model structure and loading the state dictionary into it

# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth", weights_only = True)

# This model can now be used to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"predicted: \"{predicted}\", Actual: \"{actual}\"")
