import torch
from torch.utils.data import Dataset, DataLoader
from .others.dataset import NoisyDataset
from .others.network import *
from .others.modules import *
import pickle


class Model:

    def __init__(self, model=None) -> None:
        self.device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is None:
            self.model = CustomNN(3, 3, device=self.device)
        else:
            self.model = model

    def save_pretrained_model(self, save_path="Proj_300849_302806_300585/Miniproject_2/bestmodel.pth"):
        pickle_out = open(save_path,"wb")
        pickle.dump(self.model.parameters(), pickle_out)
        pickle_out.close()

    def load_pretrained_model(self, save_path="Proj_300849_302806_300585/Miniproject_2/bestmodel.pth") -> None:
        pickle_in = open(save_path,"rb")
        model_parameters = pickle.load(pickle_in)
        for i in range(len(model_parameters)):
          self.model.parameters()[i][0].zero_()
          self.model.parameters()[i][1].zero_()
          self.model.parameters()[i][0].add_(model_parameters[i][0])
          self.model.parameters()[i][1].add_(model_parameters[i][1])

        pickle_in.close()

    def predict(self, test_input) -> torch.Tensor:
        test_in = test_input.float()
        out = self.model.forward(test_in.to(self.device))
        return out

    def train_epoch(self, loader, optimizer, loss_fn) -> None:

        for data, targets in loader:
            data = data.float().to(self.device)
            targets = targets.float().to(self.device)

            # forward
            predictions = self.model.forward(data)
            loss = loss_fn.forward(predictions, targets)

            # backward
            optimizer.zero_grad()
            self.model.backward(loss_fn.backward())
            optimizer.step()

    def train(self, train_input, train_target, optimizer=None, loss_fn=None, num_epochs=10, batch_size=100,
              learning_rate=10e-4) -> None:
        dataset = NoisyDataset(train_input, train_target)

        if loss_fn is None:
            loss_fn = MSELoss()

        if optimizer is None:
            optimizer = Adam(self.model.parameters(), device=self.device)

        for epoch in range(num_epochs):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train_epoch(dataloader, optimizer, loss_fn)
