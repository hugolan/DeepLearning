from .modules import *



class CustomNN(Module):
    def __init__(self, in_ch, out_ch, device=None):
        super(CustomNN, self).__init__()
        self.seq = Sequential(
            Conv2d(in_ch, 64, 2, stride=2, device=device),
            ReLU(device=device),
            Conv2d(64, 128, 2, stride=2, device=device),
            ReLU(device=device),
            Upsample(scale_factor=2, device=device),
            Conv2d(128, 64, 3, padding=1, device=device),
            ReLU(device=device),
            Upsample(scale_factor=2, device=device),
            Conv2d(64, out_ch, 3, padding=1, device=device),
            Sigmoid(device=device))
        self.params = self.seq.parameters()
        
    def forward(self, x):
        return self.seq.forward(x)
    
    def backward(self, grad_loss):
        return self.seq.backward(grad_loss)
    
    def parameters(self):
        return self.params
    
    def load_pretrained_model(self, model_parameters):
        for i, module in enumerate(self.seq.modules()):
            if len(model_parameters[i]) != 0: #only conv2d has paramaters
                module.load_parameters(model_parameters[i])