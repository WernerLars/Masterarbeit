from torch import nn


# Creating Standard Autoencoder Model for Spike Sorting
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, round(input_size/4)),
            #nn.ReLU(),
            nn.Linear(round(input_size/4), 2),
            #nn.ReLU(),
            #nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, round(input_size/4)),
            #nn.ReLU(),
            nn.Linear(round(input_size/4), input_size),
            #nn.ReLU(),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, round(input_size/4), 1, 1),
            nn.Conv1d(round(input_size/4), 2, 1, 1),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2, round(input_size/4), 1, 1),
            nn.ConvTranspose1d(round(input_size/4), input_size, 1, 1),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.reshape(1, self.input_size, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.reshape(1, 2, 1))
        return decoded, encoded
