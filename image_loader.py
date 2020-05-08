import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage

import torch
from torch.utils import data


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.int8(img)
    return img

def load_images(path, max_num_images = None):
    images = []
    if os.path.isfile(path):
        images.append(load_image(path))
    elif os.path.isdir(path):
        for image_name in os.listdir(path):
            images.append(load_image(path + image_name))
            if max_num_images:
                if len(images) >= max_num_images:
                    return images
    return images

def extract_image_patches(img_list, size, step, entropy_threshold):
    images = []
    positions = []
    for img in img_list:
        for crop, x, y in sliding_window(img, step, step, size):
            entropy = skimage.measure.shannon_entropy(crop)
            if entropy > entropy_threshold:
                images.append(crop)
                positions.append((x,y))
    return images, positions

def sliding_window(img, x_step, y_step, size):
    height, width, _ = img.shape
    y = 0
    while y + size < height:
        x = 0
        while x + size < width:
            yield img[y:y+size,x:x+size,:], x, y
            x += x_step
        y += y_step

class ImageDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, step=None, extract_patches = False, transform = None, entropy_threshold = 6, max_num_images = None):
        'Initialization'
        self.path = path
        self.images = load_images(path, max_num_images)
        self.positions = None
        if extract_patches:
            self.images, self.positions = extract_image_patches(self.images, 128, step, entropy_threshold)
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

  def __getitem__(self, index):
        'Generates one sample of data'
        sample = self.images[index]
        position = (0,0)
        if self.transform:
            sample = self.transform(sample)
        if self.positions:
            position = self.positions[index]
        return sample, position

class TransformedDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, ds, transform):
        'Initialization'
        self.ds = ds
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.ds)

  def __getitem__(self, index):
        'Generates one sample of data'
        sample, position = self.ds[index]
        if self.transform:
            return self.transform(sample), position
        return sample, position

        
if __name__ == '__main__':
    x = load_images('objects/T5-0/mobi/', 32, 50, 6)
    load_images('objects/T5-0/c615/', 32, 100, 6)

    print(x.shape)

    x = to_channels_last(x)

    for i in range(x.shape[0]):
        print(x[i,:,:,:].shape)
        plt.imshow(x[i,:,:,:]),plt.show()
