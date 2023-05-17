import torch.nn as nn
import torch

# Generator Code

class Generator(nn.Module):
    def __init__(self, n_classes=10, latent_dim=100, channels=3, width=64):
        super(Generator, self).__init__()


        self.layer_z = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, width * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(True)
        )

        self.layer_label = nn.Sequential(
            # input is label, going into a convolution
            nn.ConvTranspose2d(n_classes, width * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(True)
        )

        self.layer = nn.Sequential(
            # state size. ``(ngf*4) x 4 x 4``
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 8 x 8``
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            # state size. ``(ngf) x 16 x 16``
            nn.ConvTranspose2d( width, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 32 x 32``
        )

    def forward(self, noise, labels):
        x = self.layer_z(noise)
        # y = self.label_emb(labels)
        # y = y.view(y.shape[0], -1, 1, 1)
        y = self.layer_label(labels)
        x = torch.cat([x, y], 1)
        return self.layer(x)
    

class Discriminator(nn.Module):
    def __init__(self, n_classes=10, channels=3, width=64):
        super(Discriminator, self).__init__()

        self.layer_img = nn.Sequential(
            # input is img, going into a convolution
            nn.Conv2d(channels, width // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_label = nn.Sequential(
            # input is label, going into a convolution
            nn.Conv2d(n_classes, width // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer = nn.Sequential(
            # state size. ``(ndf) x 16 x 16``
            nn.Conv2d(width, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 8 x 8``
            nn.Conv2d(width * 2, width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 4 x 4``
            nn.Conv2d(width * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = self.layer_img(img)
        y = self.layer_label(labels)
        x = torch.cat([x, y], 1)
        return self.layer(x)