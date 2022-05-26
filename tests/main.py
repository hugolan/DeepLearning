from torchvision import transforms

from data_augmentation import *

from train import *
from unet import *
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    train_dir = "train_data.pkl"
    noisy_imgs_1, noisy_imgs_2 = torch.load(train_dir)
    dataset = NoisyDataset(noisy_imgs_1, noisy_imgs_2)

    aug_dataset = augment_data_rotation(dataset)

    sample_plot_dataset(aug_dataset)

    model = UNet(3, 3)
    optim = optim.Adam(model.parameters(), lr=10e-4)
    loss_fn = nn.MSELoss()

    model_outputs = train_model(input_dataset=aug_dataset,
                                load_model=False, save_model=False,
                                model=model, optimizer=optim, loss_fn=loss_fn,
                                batch_size=256, num_epochs=10)
