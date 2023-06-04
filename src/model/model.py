'''Module to create the U-NET architecture'''

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    '''
    Create a class of Double Convolutions. Takes an image (in_channel) of NxN
    pixels and pass through a convolution, and then to another convolution
    (Double Conv.)
    '''
    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()  # Why?

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 features=[64, 128, 256, 512]):

        super(UNET, self).__init__()
        # Down part of the UNET
        self.downs = nn.ModuleList()

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Max Pooling of the UNET
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Most deep layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up part of the UNET
        self.ups = nn.ModuleList()

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0],
                                    out_channels,
                                    kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Appling the Down part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Appling the bottleneck
        x = self.bottleneck(x)

        # Invert the skip connexions to facilitate the concat
        skip_connections = skip_connections[::-1]

        # Appling the Up part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # Concate along dimension 1
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert (preds.shape == x.shape)


if __name__ == "__main__":
    test()
