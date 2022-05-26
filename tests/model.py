# Make sure to only import from tqdm and not from tqdm.notebook when leaving the notebook
# for terminal
#from tqdm import tqdm
# for notebooks
import tqdm.notebook as tqdm

import torch.optim as optim
from torch.utils.data import DataLoader
from unet import *

from data_augmentation import augment_swap_data
from dataset import *




class Model():
    def __init__(self, model=None) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is None:
            self.model = UNet(3, 3).to(self.device)
        else:
            self.model = model.to(self.device)

    def load_pretrained_model(self, filename="bestmodel") -> None:
        checkpoint = torch.load(filename + ".pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_trained_model(self, filename="bestmodel") -> None:
        torch.save({'model_state_dict': self.model.state_dict()}, filename + ".pth")

    def predict(self, test_input) -> torch.Tensor:
        torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad():
            out = self.model(test_input.to(self.device))
            return out

    def train_epoch(self, loader, optimizer, loss_fn) -> None:
        scaler = None if not torch.cuda.is_available() else torch.cuda.amp.GradScaler()  # Speedup computation tricks. See https://pytorch.org/docs/stable/amp.html
        loop = tqdm(loader)

        for data, targets in loop:
            data = data.to(self.device)
            targets = targets.to(self.device)

            # forward
            if scaler is not None:
                with torch.cuda.amp.autocast():  # Speedup computation tricks. See https://pytorch.org/docs/stable/amp.html
                    predictions = self.model(data)
                    loss = loss_fn(predictions, targets)
            else:
                predictions = self.model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loop.set_postfix(loss=loss.item())

    def train(self, train_input, train_target, optimizer=None, loss_fn=None, num_epochs=10, batch_size=100,
              learning_rate=10e-4) -> None:
        torch.cuda.empty_cache()
        dataset = NoisyDataset(train_input, train_target)

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in tqdm(range(num_epochs)):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train_epoch(dataloader, optimizer, loss_fn)
            #dataset = augment_swap_data(dataset)
