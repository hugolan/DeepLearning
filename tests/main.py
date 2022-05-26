import matplotlib.pyplot as plt
from torchvision import transforms

from data_augmentation import *

from train import *
from unet import *
from transformations import *
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    train_dir = "train_data.pkl"
    noisy_imgs_1, noisy_imgs_2 = torch.load(train_dir)
    dataset = NoisyDataset(noisy_imgs_1, noisy_imgs_2)

    dataset = augment_swap_data(dataset)

    from train import *
    from unet import *
    import torch.nn as nn
    import torch.optim as optim

    model = UNet(3, 3)
    optim = optim.Adam(model.parameters(), lr=10e-4)
    loss_fn = nn.MSELoss()

    model_outputs = train_model(load_model=False, save_model=False,
                                model=model, optimizer=optim, loss_fn=loss_fn,
                                batch_size=100, num_epochs=2)

    fig = plt.figure(figsize=(12, 10))
    fig.add_subplot(2, 2, 1)
    plt.imshow(transform_for_imshow(noisy_imgs_1[5]))
    fig.add_subplot(2, 2, 2)
    plt.imshow(transform_for_imshow(dataset.X[5]))
    fig.add_subplot(2, 2, 3)
    plt.imshow(transform_for_imshow(noisy_imgs_2[5]))
    fig.add_subplot(2, 2, 4)
    plt.imshow(transform_for_imshow(dataset.Y[5]))
    plt.savefig("display")



