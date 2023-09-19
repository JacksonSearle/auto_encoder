import os
import torch
from datasets import test_dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

def visualize():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torch.load('Autoencoder.pt')
    model.to(device)

    def save_image(image, name):
        image = image.cpu().numpy()  # Move to CPU for plotting
        image = np.transpose(image, (1, 2, 0))

        if not os.path.exists('images'):
            os.makedirs('images')

        plt.imshow(image)
        plt.savefig(f'images/{name}.png')

    image = test_dataset[0][0].to(device)  # Send to the same device as the model
    print(type(image))
    save_image(image, 'test_image')

    # View a pre-determined point in latent space
    # Try changing x and y to look at different points in the latent space
    x, y = 0.0, 0.0
    result = model.decoder(torch.tensor([[x, y]], device=device))
    result = result.detach().view(1, 28, 28)
    save_image(result, 'latent_space')

    # Try changing index to different numbers
    index = 1
    image = test_dataset[index][0].to(device)  # Send to the same device as the model
    encoding, reconstruction = model(image.view(1, 784))
    reconstruction = reconstruction.cpu().detach()
    reconstruction = reconstruction.view(1, 28, 28)
    print('Original')
    save_image(image, 'original_image')
    print(f'Encoding: {encoding}\n')
    print('Reconstructed')
    save_image(reconstruction, 'reconstructed_image')

    # Displays a 2d representation of the latent space
    # The middle represents (0,0)
    # This sets how many rows and columns the table has
    step = 10
    # This sets how far apart each sampling is from the next
    size = 5
    images = []
    for i in range(step):
        for j in range(step):
            x = -size + i * (size*2)/(step-1)
            y = size - j * (size*2)/(step-1)
            result = model.decoder(torch.tensor([[x, y]], device=device))
            result = result.detach().view(28, 28)
            images.append(result)
            # print(x, y)

    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(step, step),
                     axes_pad=0.1,
                     )

    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im.cpu().numpy())  # Move to CPU for plotting

    plt.savefig('images/grid.png')

visualize()
