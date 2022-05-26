import torch
import matplotlib.pyplot as plt
from dataset import NoisyDataset
import random
from transformations import *

# for terminal
# from tqdm import tqdm
# for notebooks
import tqdm.notebook as tqdm


def sample_plot_dataset(noisy_dataset):
    x, y = noisy_dataset.X, noisy_dataset.Y
    fig = plt.figure(figsize=(12, 10))
    columns = 5
    rows = 4

    samples = random.sample(range(0, int(x.shape[0] / 2)), columns)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i < columns:
            plt.imshow(transform_for_imshow(x[samples[i]]))
        elif i < columns * 2:
            plt.imshow(transform_for_imshow(x[samples[i % columns] + int(y.shape[0] / 2)]))
        elif i < columns * 3:
            plt.imshow(transform_for_imshow(y[samples[i % columns]]))
        else:
            plt.imshow(transform_for_imshow(y[samples[i % columns] + int(y.shape[0] / 2)]))

    fig.subplots_adjust(hspace=-0.4)
    plt.savefig("display")


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

    for ind in tqdm(size[0]):
        new_x[ind] = transformation.apply(x[ind])
        new_y[ind] = transformation.apply(y[ind])

    return NoisyDataset(new_x, new_y)
