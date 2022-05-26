import torch
import matplotlib.pyplot as plt
from model import NoisyDataset
import random

# for terminal
from tqdm import tqdm


# for notebooks
# import tqdm.notebook as tqdm


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


def augment_data(noisy_dataset, transformation):
    x, y = noisy_dataset.X, noisy_dataset.Y
    size = x.shape
    new_size = (size[0] * 2, size[1], size[2], size[3])
    new_x, new_y = torch.empty(new_size), torch.empty(new_size)
    new_x[:x.shape[0]] = x[:]
    new_y[:y.shape[0]] = x[:]

    for ind in tqdm(range(x.shape[0])):
        new_x[ind + x.shape[0]] = transformation.apply(x[ind])
        new_y[ind + y.shape[0]] = transformation.apply(y[ind])

    return NoisyDataset(new_x, new_y)
