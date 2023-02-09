from torch import nn


# Creating Standard Autoencoder Model for Spike Sorting
class Autoencoder(nn.Module):
    def __init__(self, input_size, number_of_features):
        super().__init__()
        self.input_size = input_size
        self.number_of_features = number_of_features
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, round(self.input_size/4)),
            nn.LeakyReLU(),
            nn.Linear(round(self.input_size/4), self.number_of_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.number_of_features, round(self.input_size/4)),
            nn.LeakyReLU(),
            nn.Linear(round(self.input_size/4), self.input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_size, number_of_features):
        super().__init__()
        self.input_size = input_size
        self.number_of_features = number_of_features
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_size, round(self.input_size/4), 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(round(self.input_size/4), self.number_of_features, 1, 1),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.number_of_features, round(self.input_size/4), 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(round(self.input_size/4), self.input_size, 1, 1),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.reshape(1, self.input_size, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.reshape(1, self.number_of_features, 1))
        return decoded, encoded
