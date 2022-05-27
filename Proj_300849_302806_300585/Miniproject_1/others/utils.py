import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def plot_prediction_clean(img_noisy, img_clean, img_pred):
    assert img_noisy.ndim == img_clean.ndim == img_pred.ndim, "images should have same shape"
    if img_noisy.ndim == 4:
        assert img_noisy.shape[0] == img_clean.shape[0] == img_pred.shape[0] == 1, "can only plot for a single image"
        img_n = torch.squeeze(img_noisy).permute(1, 2, 0)
        img_c = torch.squeeze(img_clean).permute(1, 2, 0)
        img_p = torch.squeeze(img_pred).permute(1, 2, 0)
        img = torch.cat((img_n, img_c, img_p), dim=1)
        plt.imshow(img)
    else:
        assert img_noisy.ndim == 3, "image should be of the form nbChannels x height x width"
        img_n = img_noisy.permute(1, 2, 0)
        img_c = img_clean.permute(1, 2, 0)
        img_p = img_pred.permute(1, 2, 0)
        img = torch.cat((img_n, img_c, img_p), dim=1)
        plt.imshow(img)
        