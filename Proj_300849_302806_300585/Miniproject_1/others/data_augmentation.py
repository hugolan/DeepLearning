import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

from .dataset import NoisyDataset

from .transformations import *


def plot_aug(aug_imgs, save_path="imgs/aug_plot"):
    fig = plt.figure(figsize=(12, 5))
    columns = len(aug_imgs)
    rows = 1

    alphabet = "abcdefghijklmnopqrstuvxyz"

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(transform_for_imshow(aug_imgs[i - 1]))
        plt.title("(" + alphabet[i - 1] + ")")

    plt.savefig(save_path)
    plt.show()


def plot_swap(aug_imgs, save_path="imgs/aug_swap"):
    fig = plt.figure(figsize=(12, 5))
    columns = len(aug_imgs)
    rows = 2

    alphabet = "abcdefghijklmnopqrstuvxyz"

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i <= columns:
            plt.imshow(transform_for_imshow(aug_imgs[(i - 1) % len(aug_imgs)][0]))
        else:
            plt.imshow(transform_for_imshow(aug_imgs[(i - 1) % len(aug_imgs)][1]))
        plt.title("(" + alphabet[i - 1] + ")")

    plt.savefig(save_path)
    plt.show()


def transform_for_imshow(img):
    return (img.float() / 255.0).permute(1, 2, 0)


def augment_swap_data(noisy_dataset):
    x, y = noisy_dataset.X, noisy_dataset.Y
    size = x.shape
    new_x, new_y = torch.empty(size), torch.empty(size)

    for ind in tqdm(range(size[0])):
        new_x[ind], new_y[ind] = swap_aug(x[ind], y[ind])

    return NoisyDataset(new_x, new_y)


def augment_data(noisy_dataset, transformation):
    x, y = noisy_dataset.X, noisy_dataset.Y
    size = x.shape
    new_x, new_y = torch.empty(size), torch.empty(size)

    for ind in tqdm(range(size[0])):
        new_x[ind], new_y[ind] = transformation.apply(x[ind], y[ind])

    return NoisyDataset(new_x, new_y)
