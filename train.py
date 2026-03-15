import os
import random

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ==============================
# Set Seeds (Reproducibility)
# ==============================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print("PyTorch version:", torch.__version__)


# ==============================
# Load MNIST Dataset
# ==============================

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

BATCH_SIZE = 64

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ==============================
# Generator Model
# ==============================

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1, 28, 28)


# ==============================
# Discriminator Model
# ==============================

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


generator = Generator()
discriminator = Discriminator()


# ==============================
# Loss and Optimizers
# ==============================

criterion = nn.BCELoss()
learning_rate = 0.1
EPOCHS = 1

generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)


# ==============================
# MLflow Experiment
# ==============================

mlflow.set_experiment("Assignment3_Nosyba")

with mlflow.start_run():

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.set_tag("student_id", "YOUR_ID")

    NOISE_DIM = 100

    # ==============================
    # Training Loop
    # ==============================

    for epoch in range(EPOCHS):

        gen_loss_epoch = 0
        disc_loss_epoch = 0
        num_batches = 0

        for real_images, _ in dataloader:

            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            noise = torch.randn(batch_size, NOISE_DIM)
            fake_images = generator(noise)

            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images.detach())

            disc_loss = (
                criterion(real_output, real_labels)
                + criterion(fake_output, fake_labels)
            )

            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            discriminator_optimizer.step()

            fake_output = discriminator(fake_images)
            gen_loss = criterion(fake_output, real_labels)

            generator_optimizer.zero_grad()
            gen_loss.backward()
            generator_optimizer.step()

            gen_loss_epoch += gen_loss.item()
            disc_loss_epoch += disc_loss.item()
            num_batches += 1

        avg_g_loss = gen_loss_epoch / num_batches
        avg_d_loss = disc_loss_epoch / num_batches

        print(
            f"Epoch {epoch + 1} Generator Loss: {avg_g_loss:.4f} "
            f"Discriminator Loss: {avg_d_loss:.4f}"
        )

        mlflow.log_metric("generator_loss", avg_g_loss, step=epoch)
        mlflow.log_metric("discriminator_loss", avg_d_loss, step=epoch)


    # ==============================
    # Save Models in MLflow
    # ==============================

    mlflow.pytorch.log_model(generator, "generator_model")
    mlflow.pytorch.log_model(discriminator, "discriminator_model")


# ==============================
# Save MNIST as CSV
# ==============================

x_data = dataset.data.numpy().reshape(-1, 28 * 28)
df = pd.DataFrame(x_data)
df.to_csv("mnist_data.csv", index=False)


# ==============================
# Generate Sample Image
# ==============================

noise = torch.randn(1, 100)
generated_image = generator(noise).detach()

plt.imshow(generated_image[0][0], cmap="gray")
plt.axis("off")
plt.title("Generated Image")
plt.show()

print(mlflow.list_experiments())