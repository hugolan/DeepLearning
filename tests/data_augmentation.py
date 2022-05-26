import torch
import matplotlib.pyplot as plt
from model import NoisyDataset
import random
from torchvision import transforms
import torchvision.transforms.functional as TF


def sample_plot_dataset(noisy_dataset):
    x, y = noisy_dataset.X, noisy_dataset.Y
    fig = plt.figure(figsize=(15, 8))
    columns = 5
    rows = 2

    samples = random.sample(range(int(x.shape[0] / 2), x.shape[0]), columns)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i < columns:
            plt.imshow(transform_for_imshow(x[samples[i]]))
        else:
            plt.imshow(transform_for_imshow(y[samples[i % columns]]))

    fig.subplots_adjust(hspace=-0.4)
    plt.savefig("display")


def transform_for_imshow(img):
    return (img.float() / 255.0).permute(1, 2, 0)


# ColorJitter
# RandomHorizontalFlip
# RandomPerspective
# RandomVerticalFlip
# GaussianBlur
# RandomInvert

def augment_data_rotation(noisy_dataset):
    return augment_data(noisy_dataset, TF.rotate, 180)


def augment_data_color(noisy_dataset):
    return augment_data(noisy_dataset, TF.adjust_hue, 0.5

                        )


def augment_data_blur():
    pass


def augment_data(noisy_dataset, transforms_apply, args):
    x, y = noisy_dataset.X, noisy_dataset.Y
    size = x.shape
    new_size = (size[0] * 2, size[1], size[2], size[3])
    new_x, new_y = torch.empty(new_size), torch.empty(new_size)
    new_x[:x.shape[0]] = x[:]
    new_y[:y.shape[0]] = x[:]

    for ind in range(x.shape[0]):
        new_x[ind + x.shape[0]] = transforms_apply(x[ind],args)
        new_y[ind + y.shape[0]] = transforms_apply(y[ind], args)

    return NoisyDataset(new_x, new_y)
