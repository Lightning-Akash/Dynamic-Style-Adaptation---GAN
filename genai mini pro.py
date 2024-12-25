import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
batch_size = 64
image_size = 64
latent_dim = 100
epochs = 35
learning_rate = 0.0002
beta1 = 0.5
output_dir = "output_images"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Data transformation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# CIFAR-10 dataset
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root="cifar10_data", download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Generator definition
class Generator(nn.Module):
    def __init__(self, latent_dim):  # Corrected __init__ method
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self):  # Corrected __init__ method
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Device configuration (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))


# Training function
def train():
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    for epoch in range(epochs):
        for i, (data, _) in enumerate(dataloader):

            # Update Discriminator
            discriminator.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1.0, device=device)

            # Real data loss
            output = discriminator(real_data).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()

            # Generate fake data
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_data = generator(noise)
            label.fill_(0.0)

            # Fake data loss
            output = discriminator(fake_data.detach()).view(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()

            optimizerD.step()

            # Update Generator
            generator.zero_grad()
            label.fill_(1.0)
            output = discriminator(fake_data).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            optimizerG.step()

            # Print training progress
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{i}/{len(dataloader)}] Loss D: {lossD_real.item() + lossD_fake.item():.4f}, Loss G: {lossG.item():.4f}"
                )

        # Save images at the end of each epoch
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
            grid = utils.make_grid(fake_images, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.savefig(f"{output_dir}/epoch_{epoch}.png")
            plt.close()


# Start training
train()
# Save the trained Generator and Discriminator
torch.save(generator.state_dict(), 'generator.pth')  # Save Generator weights
torch.save(discriminator.state_dict(), 'discriminator.pth')  # Save Discriminator weights
