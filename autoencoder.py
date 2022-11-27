from torch import nn


# Creating Autoencoder Model for Spike Sorting
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, round(input_size/2)),
            nn.ReLU(),
            nn.Linear(round(input_size/2), 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, round(input_size/2)),
            nn.ReLU(),
            nn.Linear(round(input_size/2), input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
