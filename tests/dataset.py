from torch.utils.data import Dataset


class NoisyDataset(Dataset):
    def __init__(self, X, Y):
        assert X.shape == Y.shape and X.ndim == 4
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
