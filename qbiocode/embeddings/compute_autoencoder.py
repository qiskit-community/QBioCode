"""Convolutional autoencoder embedding.

Importing this module maps torch -- and so torch's copy of ``libomp`` -- into the
process. On macOS that must not happen before xgboost's copy is initialised or
XGBoost model fitting segfaults; see ``qbiocode.utils._openmp`` for the full
diagnosis. The preload below makes this module safe to import on its own, in any
order, rather than relying on the caller having imported ``qbiocode`` first.
"""

from qbiocode.utils._openmp import preload_openmp_libraries

preload_openmp_libraries()

import torch
import torch.nn as nn
import torch.optim as optim


# Define the Autoencoder Model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, stride=2, padding=1),  # (64, 192, 192)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 96, 96)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 48, 48)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (512, 24, 24)
            nn.ReLU(),
            nn.Conv2d(512, 7, kernel_size=3, stride=2, padding=1),  # (7, 16, 16)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                7, 512, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (512, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (256, 48, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (128, 96, 96)
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (64, 192, 192)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 7, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (7, 384, 384)
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
