import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from model import Model
from datasets import train_loader, val_loader, test_loader

encoding_dim = 2
input_dim = 28*28 # 784
label_dim = 10
batch_size = 512
epochs = 100

model_layers = [input_dim, 200, 50, 14, encoding_dim]

# Initialize the neural network with specified layers
model = Model(model_layers)
# Put the neural network on the GPU (if available) for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Learning Rate set
lr = 1e-3
# Optimizer helps with gradient descent
optimizer = optim.Adam(model.parameters(), lr=lr)
# Mean Square Error is commonly used for output value differences
loss_function = nn.MSELoss()

def calculate_error(model, data_loader):
    total_error = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.view(-1, 784).to(device)
            _, reconstructed = model(images)
            reconstruction_loss = loss_function(reconstructed, images)
            total_error += reconstruction_loss
    return total_error

epochs = 30
for epoch in trange(epochs):
    total_loss = 0
    for batch_features, batch_labels in train_loader:

        # Resize batch_features
        batch_features = batch_features.view(-1, 784).to(device)

        # Reset the gradients back to zero
        model.zero_grad()

        # Run examples through generator
        _, reconstructed = model(batch_features)

        # Compute reconstruction loss
        reconstruction_loss = loss_function(reconstructed, batch_features)
        reconstruction_loss.backward()
        optimizer.step()

        # Calculate accuracy on training set
        total_loss += reconstruction_loss

    # Calculate and print accuracy on validation set
    val_error = calculate_error(model, val_loader)
    print("Epoch: {}/{} | Train Loss: {:.4f} | Val Acc: {:.4f}".format(epoch + 1, epochs, total_loss, val_error))

# Test the model on the test set
test_error = calculate_error(model, test_loader)
print("\nTest Accuracy: {:.4f}".format(test_error))

# Save the model
torch.save(model, "Autoencoder.pt")