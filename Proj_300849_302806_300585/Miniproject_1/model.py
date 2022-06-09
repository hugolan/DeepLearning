# Make sure to only import from tqdm and not from tqdm.notebook when leaving the notebook
from tqdm.notebook import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .others.dataset import NoisyDataset
from .others.unet import *
from .others.data_augmentation import augment_swap_data

class Model:

    def __init__(self, model=None) -> None:
        self.device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is None:
            self.model = UNet(3, 3).to(self.device)
        else:
            self.model = model.to(self.device)

    def load_pretrained_model(self, filename="Proj_300849_302806_300585/Miniproject_1/bestmodel.pth") -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_trained_model(self, filename="Proj_300849_302806_300585/Miniproject_1/bestmodel.pth") -> None:
        torch.save({'model_state_dict': self.model.state_dict()}, filename)

    def predict(self, test_input) -> torch.Tensor:
        torch.cuda.empty_cache()
        test_in = test_input.float()/255.0
        self.model.eval()
        with torch.no_grad():
            out = self.model(test_in.to(self.device))
            out = out * 255
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
              learning_rate=10e-4, dataset_swap=False) -> None:
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
            if dataset_swap and epoch != num_epochs - 1:
                dataset_tmp = augment_swap_data(dataset)
                del dataset.X, dataset.Y
                del dataset
                dataset = dataset_tmp

        del dataset.X, dataset.Y
        del dataset