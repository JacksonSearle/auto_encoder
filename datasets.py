import torch
import torchvision
from torchvision import datasets

# MNIST is already normalized, so we won't normalize here
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Obtain train_dataset
train_dataset = datasets.MNIST(
   root="./mnist",
   train=True,
   transform=transform,
   download=True,
)

# Obtain test_dataset
test_dataset = datasets.MNIST(
   root="./mnist",
   train=False,
   transform=transform,
   download=True,
)


# Split the training dataset into training and validation sets
# 80% train, 20% validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# These loaders help us easily acces the data when training
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)