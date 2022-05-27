import torch
from torch.utils.data import Dataset, DataLoader
from others.dataset import NoisyDataset
from others.network import *
from others.modules import *

class Model:

    def __init__(self, model=None) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is None:
            self.model = CustomNN(3, 3, device=self.device)
        else:
            self.model = model

    def load_pretrained_model(self, filename="bestmodel") -> None:
        checkpoint = torch.load(filename + ".pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_trained_model(self, filename="bestmodel") -> None:
        torch.save({'model_state_dict': self.model.state_dict()}, filename + ".pth")

    def predict(self, test_input) -> torch.Tensor:
        test_in = test_input.float()/255.0
        out = self.model.forward(test_in.to(self.device))
        out = out * 255
        return out

    def train_epoch(self, loader, optimizer, loss_fn) -> None:

        for data, targets in loader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            # forward
            predictions = self.model.forward(data)
            loss = loss_fn.forward(predictions, targets)

            # backward
            optimizer.zero_grad()
            model.backward(loss_fn.backward())
            optimizer.step()

    def train(self, train_input, train_target, optimizer=None, loss_fn=None, num_epochs=10, batch_size=100,
              learning_rate=10e-4) -> None:
        dataset = NoisyDataset(train_input, train_target)

        if loss_fn is None:
            loss_fn = MSELoss()

        if optimizer is None:
            optimizer = Adam(self.model.parameters(), lr=learning_rate, device=self.device)

        for epoch in range(num_epochs):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train_epoch(dataloader, optimizer, loss_fn)