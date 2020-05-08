import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # ENCODER
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv4a = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.conv4b = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv4c = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.conv4d = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.conv4e = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.conv4f = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        #self.fc1  = nn.Linear(4*4*128, 3072*3)
        self.fc21 = nn.Linear(4*4*128, 3072 * 8)
        self.fc22 = nn.Linear(4*4*128, 3072 * 8)

        #DECODER
        self.dropout = nn.Dropout(0.5)

        #self.fc3 = nn.Linear(3072 * 3, 3072 * 3)
        self.fc4 = nn.Linear(3072 * 8, 4*4*128)
        self.conv5e = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.conv5f = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.conv5a = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.conv5b = nn.Conv2d(128, 64, kernel_size = 3, padding = 1)
        self.conv5c = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv5d = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(16, 3, kernel_size = 3, padding = 1)

    def encode(self, x):
        # 3 x 32 x 32 = 3072

        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2)
        # 32 x 16 x 16 = 8192

        h = self.conv3(h)
        h = F.relu(h)
        h = self.conv4(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2)
        # 64 x 8 x 8 = 4096

        h = self.conv4a(h)
        h = F.relu(h)
        h = self.conv4b(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2)
        # 64 x 8 x 8 = 4096

        h = self.conv4c(h)
        h = F.relu(h)
        h = self.conv4d(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2)

        h = self.conv4e(h)
        h = F.relu(h)
        h = self.conv4f(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2)

        h = h.view(-1, 4*4*128)
        # 4096

        #h = self.fc1(h)
        #h = F.relu(h)
        # 512

        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #512
        h = self.dropout(z)

        #h = self.fc3(z)
        #h = F.relu(h)
        #4096

        h = self.fc4(h)
        h = F.relu(h)
        #1024

        h = h.view(-1, 128, 4, 4)
        # 64 x 8 x 8 = 1024

        h = F.upsample(h, scale_factor=2)
        h = self.conv5e(h)
        h = F.relu(h)
        h = self.conv5f(h)
        h = F.relu(h)


        h = F.upsample(h, scale_factor=2)
        h = self.conv5a(h)
        h = F.relu(h)
        h = self.conv5b(h)
        h = F.relu(h)

        h = F.upsample(h, scale_factor=2)
        h = self.conv5c(h)
        h = F.relu(h)
        h = self.conv5d(h)
        h = F.relu(h)
        # 8 x 16 x 16 = 2048

        h = F.upsample(h, scale_factor=2)
        h = self.conv5(h)
        h = F.relu(h)
        h = self.conv6(h)
        h = F.relu(h)
        # 8 x 16 x 16 = 2048

        h = F.upsample(h, scale_factor=2)
        h = self.conv7(h)
        h = F.relu(h)
        h = self.conv8(h)
        # 3 x 32 x 32 = 3072

        return torch.sigmoid(h)

    def forward(self, x):
        #mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
