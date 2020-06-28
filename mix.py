import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np

from image_loader import *
from model import VAE

import os

parser = argparse.ArgumentParser(description='Image Mixer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model', type=str, default='model.pt',
                    help='which model to use')
parser.add_argument('--images', nargs=2, type=str, default=(None,None),
                    help='images to mix')
parser.add_argument('--enable-matplotlib', action='store_true', default=False,
                    help='enables showing the mixing result with matplotlib')
args = parser.parse_args()

if args.enable_matplotlib:
    import matplotlib.pyplot as plt

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

image1_dataset =\
        ImageDataset(args.images[0], transform=transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.CenterCrop(128),
    transforms.ToTensor()
    ]))

image1_loader = torch.utils.data.DataLoader(image1_dataset,
    batch_size=1, shuffle=False, **kwargs)

image2_dataset =\
        ImageDataset(args.images[1], transform=transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.CenterCrop(128),
    transforms.ToTensor()
    ]))

image2_loader = torch.utils.data.DataLoader(image2_dataset,
    batch_size=1, shuffle=False, **kwargs)

if __name__ == "__main__":
    print('Using model: {}'.format(args.model))
    model = torch.load(args.model)
    model.eval()

    N_TEST_IMG = 12

    if args.enable_matplotlib:
        f, a = plt.subplots(1, N_TEST_IMG+2, figsize=(32, 32))
        plt.axis('off')

    try:
        os.mkdir('result')
    except:
        pass

    with torch.no_grad():
        img1 = next(iter(image1_loader))[0].cuda()
        img2 = next(iter(image2_loader))[0].cuda()

        if args.enable_matplotlib:
            a[0].imshow(np.moveaxis(img1[0].data.cpu().numpy(), 0, 2))
            a[-1].imshow(np.moveaxis(img2[0].data.cpu().numpy(), 0, 2))
            a[0].set_xticks(()); a[0].set_yticks(())

        encoded1, _ = model.encode(img1)
        encoded2, _ = model.encode(img2)

        def crossover (x, y, n):
            out = torch.tensor(x)
            out[:n] = y[:n]
            return out

        for i in range(N_TEST_IMG):
            new_encoding = crossover(encoded1[0], encoded2[0], int(8*3072 * i / (N_TEST_IMG-1)))
            decoded_data = model.decode(new_encoding.unsqueeze(0))[0]

            img_copy = torch.Tensor(decoded_data.cpu())
            save_image(img_copy, 'result/{}.png'.format(i))

            if args.enable_matplotlib:
                img = (np.moveaxis(decoded_data.data.cpu().numpy(), 0, 2))
                a[i+1].imshow(img)
                a[i+1].set_xticks(()); a[i+1].set_yticks(())

        if args.enable_matplotlib:
            plt.draw(); plt.pause(0.05)
            plt.show()
