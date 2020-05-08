import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np

from model import VAE

parser = argparse.ArgumentParser(description='Train a new image mixing model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset', type=str, default=None,
                    help='dataset to use for training')
parser.add_argument('--max-num-images', type=int, default=90000,
                    help='maximum number of images from the training set to use')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}

image_dataset =\
        datasets.ImageFolder(
                root=args.dataset,
                transform=transforms.Compose([
    transforms.CenterCrop(128),
    transforms.ToTensor()
    ]))

image_dataset = torch.utils.data.Subset(image_dataset, range(args.max_num_images))
train_size = int(0.8 * len(image_dataset))
test_size = len(image_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, milestones=[3], gamma=0.1)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    p = 0.8
    return p * BCE + (1 - p) * KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    scheduler.step()


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss

if __name__ == "__main__":
    N_TEST_IMG = 6

    min_test_loss = float('inf')
    num_epochs_above_min = 0

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test(epoch)
        with torch.no_grad():
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                num_epochs_above_min = 0
                torch.save(model, 'model.pt')
            else:
                num_epochs_above_min += 1
                if num_epochs_above_min >= 10:
                    print('Training finished, halting.')
                    break
