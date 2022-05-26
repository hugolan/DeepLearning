import torch
import matplotlib.pyplot as plt
from torchvision import transforms

if __name__ == "__main__":
    train_dir = "train_data.pkl"
    noisy_imgs_1, noisy_imgs_2 = torch.load("train_data.pkl")

    image_ind = 65

    plt.imshow((noisy_imgs_1[image_ind].float() / 255.0).permute(1, 2, 0))
    plt.savefig("display_before")

    #ColorJitter
    #RandomApply
    #RandomHorizontalFlip
    #RandomPerspective
    #RandomVerticalFlip
    #GaussianBlur
    #RandomInvert
    #RandomResizedCrop

    augmentation = transforms.Compose([
        transforms.RandomRotation((90, 90)),
        transforms.RandomInvert(),
        transforms.RandomVerticalFlip(),
    ])
    for ind in range(noisy_imgs_1.shape[0]):
        augmentation(noisy_imgs_1[ind])

    plt.imshow((augmentation(noisy_imgs_1[image_ind]).float() / 255.0).permute(1, 2, 0))
    plt.savefig("display_after")
