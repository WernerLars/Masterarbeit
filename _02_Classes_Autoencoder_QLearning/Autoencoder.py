import torch
from torch import nn
from torchview import draw_graph


# Creating Standard Autoencoder Model for Spike Sorting
class Autoencoder(nn.Module):
    def __init__(self, input_size, number_of_features):
        super().__init__()
        self.input_size = input_size
        self.number_of_features = number_of_features
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, round(self.input_size / 4)),
            nn.LeakyReLU(),
            nn.Linear(round(self.input_size / 4), self.number_of_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.number_of_features, round(self.input_size / 4)),
            nn.LeakyReLU(),
            nn.Linear(round(self.input_size / 4), self.input_size),
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
            nn.Conv1d(in_channels=self.input_size, out_channels=round(self.input_size/4),
                      kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=round(self.input_size/4), out_channels=self.number_of_features,
                      kernel_size=1),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.number_of_features, out_channels=round(self.input_size/4),
                               kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=round(self.input_size/4), out_channels=self.input_size,
                               kernel_size=1),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.reshape(1, self.input_size, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.reshape(1, self.number_of_features, 1))
        return decoded, encoded


class ConvolutionalAutoencoderTest(nn.Module):
    def __init__(self, input_size, number_of_features):
        super().__init__()
        self.input_size = input_size
        self.number_of_features = number_of_features
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=30),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=17),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=17),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=30),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.reshape(1, 1, self.input_size)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded.reshape(1, 1, self.number_of_features))
        return decoded, encoded


def printAutoencoderModel():
    spike = torch.tensor([-2.11647214e-02, -2.00144278e-02, -2.48166304e-02, -2.70972753e-02,
                          -1.11241704e-02, 1.86904987e-02, 3.99716833e-02, 4.40400999e-02,
                          4.38833221e-02, 5.06364129e-02, 6.02243042e-02, 3.59622148e-02,
                          -9.64451652e-02, -3.71359573e-01, -6.92987060e-01, -8.74449953e-01,
                          -7.13363902e-01, -1.84182190e-01, 4.08997970e-01, 7.26119515e-01,
                          7.19977210e-01, 5.61000789e-01, 4.04007238e-01, 2.96025242e-01,
                          2.22861462e-01, 1.69209408e-01, 1.33269005e-01, 1.11481721e-01,
                          9.67043158e-02, 8.35988040e-02, 6.87571423e-02, 5.74871826e-02,
                          5.26722178e-02, 4.53956038e-02, 3.31356602e-02, 2.21250606e-02,
                          1.35048482e-02, -4.41592673e-04, -2.31921908e-02, -4.69576347e-02,
                          -6.03503288e-02, -6.27551095e-02, -6.19812766e-02, -6.37499251e-02,
                          -6.42747873e-02, -5.93586264e-02, -5.06150772e-02])
    #model_graph = draw_graph(ConvolutionalAutoencoder(len(spike), 2), input_data=spike)
    #model_graph.visual_graph.render(format="png")
    torch.onnx.export(Autoencoder(len(spike), 2), args=spike, f="autoencoder.onnx",
                      input_names=["Original Spike"],
                      output_names=["Reconstructed Spike", "Features"])
    torch.onnx.export(ConvolutionalAutoencoder(len(spike), 2), args=spike, f="conv.onnx",
                      input_names=["Original Spike"],
                      output_names=["Reconstructed Spike", "Features"])


#printAutoencoderModel()


