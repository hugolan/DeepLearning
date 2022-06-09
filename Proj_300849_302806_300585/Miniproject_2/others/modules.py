import torch
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math

class Module(object):
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
    
class Optim(object):
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
        raise NotImplementedError
        
class Conv2d(Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device=None):
        super(Conv2d, self).__init__()
        self.id = 'Conv2d'
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        mean = 0
        std = math.sqrt(2.0 / (kernel_size*kernel_size*out_channels))
        self.weight = torch.empty((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float).normal_(mean=mean, std=std)
        self.grad_weight = torch.empty((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float).zero_()    
        self.bias = None
        self.grad_bias = None
        if bias:
            self.bias = torch.empty((out_channels), dtype=torch.float).normal_(mean=mean, std=std)
            self.grad_bias = torch.empty((out_channels), dtype=torch.float).zero_()
        if device is not None:
            self.weight = self.weight.to(device)
            self.grad_weight = self.grad_weight.to(device)
            self.bias = self.bias.to(device)
            self.grad_bias = self.grad_bias.to(device)

    
    def forward(self, input_t):
        assert input_t.ndim == 4 and input_t.shape[1] == self.in_channels, f"input tensor should be of shape (nImages, {self.in_channels}, Y, X)"

        self.in_size_y = input_t.shape[2]
        self.in_size_x = input_t.shape[3]
        
        #General formula for computing the output sizes of images depending on the input size, the kernel size, the padding and the stride.
        self.out_size_y = ((self.in_size_y + 2*self.padding - self.kernel_size) // self.stride) + 1
        self.out_size_x = ((self.in_size_x + 2*self.padding - self.kernel_size) // self.stride) + 1
        
        #Here we use unfold to facilitate the sliding window operations of our convolution
        # unfold takes an input a tensor of shape (N, C, Y, X) where N is the number of 2D images, C is the number of channels per image, 
        # Y, X is the dimension of each 2D image. Unfold also takes as input the kernel size K (in our convolution we assume the kernel to always be a square, thus number of elements in kernel is K*K).
        # For each image, unfold extracts patches of same size as the kernel size, for each channel.
        # Each patch can be seen a 1D vertical vector (instead of a 2D matrix), and patches of the same channel are concatenated so as to obtain, for one channel, a 2D matrix of shape (K*K, out_size_y*out_size_x). Patches are concatenated in order from left to right and from top to bottom.
        # Then matrices of each channel are concatenated vertically so as to obtain a 2D matrix of shape (C*K*K, out_size_y*out_size_x). Matrices are concatenated in order of channels.
        # out_size_y represents the Y axis of output image, out_size_x represents the X axis of output image.
        # Since this procedure is done for each image, the resulting tensor of unfold is of shape (N, C*K*K, out_size_y*out_size_x).
        patch_tensor = torch.nn.functional.unfold(input_t, (self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride)
        
        #Useful for backward pass
        self.input_unfolded = patch_tensor
        
        # Transpose the patches tensor so as to obtain, for each image, the C patches of same location in a row instead of in a column.
        patch_tensor = patch_tensor.transpose(1, 2)
        
        #Reshape the kernel so as to match the shape of patches in patches tensor. Note that each row in the patches tensor is of size C*K*K,
        # and each row in the reshaped kernel is also of size (C*K*K).
        kernel_w = self.weight.view(self.out_channels, -1)
        
        #For matrix multiplication (apply the kernel on the patches) of the patches tensor with the reshaped kernel, we transpose the reshaped kernel so as to have C*K*K rows.
        kernel_w = kernel_w.t()
        
        #Matrix multiplication, i.e. apply kernel on patches
        #Resulting conv has shape (N, out_size_y*out_size_x, out_channels)
        conv = patch_tensor @ kernel_w
        
        #Transpose and reshape to match the input tensor shape (batch of images), get shape (N, out_channels, out_size_y, out_size_x)
        output_t = conv.transpose(1, 2).reshape(input_t.shape[0], self.out_channels, self.out_size_y, self.out_size_x)
        
        #Add bias if needed
        if self.bias is not None:
            output_t += self.bias.view(1, self.out_channels, 1, 1)
        
        return output_t
        
    def backward(self, grad_output):
        assert grad_output.ndim == 4 and grad_output.shape[1] == self.out_channels and grad_output.shape[2] == self.out_size_y and grad_output.shape[3] == self.out_size_x, f"input tensor should be of shape (nImages, {self.out_channels}, {self.out_size_y}, {self.out_size_x})"
        
        #Reshape into (N, out_channels, out_size_y*out_size_x)
        grad_out_reshaped = grad_output.reshape(grad_output.shape[0], self.out_channels, -1)
        
        #Transpose to pass channels in last dimension to get (N, out_size_y*out_size_x, out_channels)
        grad_out_reshaped = grad_out_reshaped.transpose(1, 2)
        
        #Compute gradient of loss wrt input using the kernel weights. Resulting shape (N, out_size_y*out_size_x, C*K*K).
        #C = A@B => d_l/d_A = d_l/d_C @ B.t, out = in@weights => d_l/d_in = d_l/d_out @ weights
        grad_input_unfolded = grad_out_reshaped @ self.weight.view(self.out_channels, -1)
        
        #Reshape gradient of loss wrt input by "unpatching" the images to obtain a (N, C, in_size_y, in_size_x) tensor
        grad_input = torch.nn.functional.fold(grad_input_unfolded.transpose(1, 2), (self.in_size_y, self.in_size_x),(self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride)
        assert grad_input.shape == (grad_output.shape[0], self.in_channels, self.in_size_y, self.in_size_x)
        
        #Compute gradient of loss wrt weight kernel. Resulting shape (N, C*K*K, out_channels)
        #C = A@B => d_l/d_B = A.t @ d_l/d_C, since out=in@w (unfolded), d_l/d_w = in.t @ d_l/d_out
        grad_weights = self.input_unfolded @ grad_out_reshaped
        
        #Sum over the batch of N images (sum over the gradients of images). Resulting shape (C*K*K, out_channels)
        grad_weights = grad_weights.sum(dim=0)
        
        #Reshape to obtain original weights shape (out_channels, in_channels (C), K, K)
        self.grad_weight.add_(grad_weights.t().view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        
        if self.bias is not None:
            #For each image, sum over full image to obtain 1 bias value for each "out_channel" channel.
            #Sum over the batch of N images (sum over the gradients of images). Resulting shape (out_channels).
            self.grad_bias.add_(grad_output.sum(dim=[0, 2, 3]))
        return grad_input
        
    def parameters(self):
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]
    
    def load_parameters(self, parameters):
        (self.weight, self.grad_weight), (self.bias, self.grad_bias) = parameters[0], parameters[1]
    
    
class Upsample(Module):
    
    def __init__(self, scale_factor=1, device=None):
        super(Upsample, self).__init__()
        assert isinstance(scale_factor, int) and scale_factor > 0, "scale factor should be a positive integer"
        self.id = 'Upsample'
        self.device = device
        self.scale_factor = scale_factor
    
    def forward(self, input_t):
        #self.input_t = input_t # to get shape later on
        first_upsample = input_t.repeat_interleave(self.scale_factor, dim=2)
        second_upsample = first_upsample.repeat_interleave(self.scale_factor, dim=3)
        return second_upsample
        
    def backward(self, grad_output): # basically 'num_channels' convolutions
        in_size_y = grad_output.shape[2]
        in_size_x = grad_output.shape[3]
        
        out_size_y = ((in_size_y - self.scale_factor) // self.scale_factor) + 1
        out_size_x = ((in_size_x - self.scale_factor) // self.scale_factor) + 1
        
        patch_tensor = torch.nn.functional.unfold(grad_output, (self.scale_factor, self.scale_factor), padding=0, stride=self.scale_factor)
        patch_tensor = patch_tensor.reshape((grad_output.shape[0], grad_output.shape[1], self.scale_factor*self.scale_factor, -1))
        output_t = patch_tensor.sum(2).reshape((grad_output.shape[0], grad_output.shape[1], out_size_y, out_size_x))
        
        return output_t
        
    def parameters(self): #no params on upsampling layers
        return []
    

class Sigmoid(Module):
    
    def __init__(self, device=None):
        super(Sigmoid, self).__init__()
        self.id = 'Sigmoid'
        self.device = device
        
    def forward(self, input_t):
        self.sig = input_t.sigmoid()
        return self.sig
        
    #grad_output is d_l/d_x and we want to compute d_l/d_s
    def backward(self, grad_output):
        #sig*(1-sig)
        grad_sig = self.sig.mul(1-self.sig)
        grad_sig = grad_output.mul(grad_sig)
        return grad_sig
        
    def parameters(self):
        return []
    
    
class ReLU(Module):
    
    def __init__(self, device=None):
        super(ReLU, self).__init__()
        self.id = 'ReLU'
        self.device = device

    def forward(self, input_t):
        #max(0, x)
        torch_zeros = torch.empty(input_t.shape).fill_(0)
        if self.device is not None:
            torch_zeros = torch_zeros.to(self.device)
        self.relu = input_t.maximum(torch_zeros)
        return self.relu
        
        
    def backward(self, grad_output):
        #sign(max(0, x))
        grad_relu = self.relu.gt(0).int().float()
        return grad_output.mul(grad_relu)
        
    def parameters(self):
        return []
    

class Sequential(Module):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.id = 'Sequential'
        self.modules = [m for m in modules]

    def modules():
        return self.modules
    
    def forward(self, input_t):
        x = input_t.clone()
        for i in range(len(self.modules)):
            x = self.modules[i].forward(x)
        return x
        
    def backward(self, grad_output):
        x = grad_output.clone()
        for i in reversed(range(len(self.modules))):
            x = self.modules[i].backward(x)
        return x
        
    def parameters(self):
        params = []
        for module in self.modules:
            if module.id == 'Sequential':
                for seq_module in module.parameters():
                    params = params + seq_module.parameters()
            elif module.parameters():
                params = params + module.parameters()
        return params
    
    
class MSELoss(Module):
    
    def __init__(self):
        super(MSELoss, self).__init__()
        pass
    
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return (pred - target).square().mean()
        
    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.numel()
        
    def parameters(self):
        return []
    
    
class SGD(Optim):
    def __init__(self, model_params, lr=0.001, device=None):
        super(SGD, self).__init__()
        self.model_params = model_params
        self.lr = lr
        
    def step(self):
        for w, grad_w in self.model_params:
            if w is not None and grad_w is not None:
                w.sub_(self.lr*grad_w)

    def zero_grad(self):
        for w, grad_w in self.model_params:
            if w is not None and grad_w is not None:
                grad_w.zero_()
                
                
class Adam(Optim):
    def __init__(self, model_params, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-15, device=None):
        super(Adam, self).__init__()
        self.m_dw = {}
        self.v_dw = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.model_params = model_params
        self.t = 1
        for i in range(len(self.model_params)):
            self.m_dw[i] = torch.empty(self.model_params[i][1].shape).zero_()
            self.v_dw[i] = torch.empty(self.model_params[i][1].shape).zero_()
            if device is not None:
                self.m_dw[i] = self.m_dw[i].to(device)
                self.v_dw[i] = self.v_dw[i].to(device)
        

    def step(self):
        for i in range(len(self.model_params)):
            if self.model_params[i][0] is not None and self.model_params[i][1] is not None:
                curr_mean = self.m_dw[i]
                curr_speed = self.v_dw[i]
                
                self.m_dw[i] = self.beta1*curr_mean + (1-self.beta1)*self.model_params[i][1]

                self.v_dw[i] = self.beta2*curr_speed + (1-self.beta2)*(self.model_params[i][1].square())

                m_estimate = self.m_dw[i]/(1-(self.beta1**self.t))
                v_estimate = self.v_dw[i]/(1-(self.beta2**self.t))

                self.model_params[i][0].sub_(self.eta*(m_estimate/(v_estimate.sqrt()+self.epsilon)))

        self.t = self.t + 1

    def zero_grad(self):
        for i in range(len(self.model_params)):
            if self.model_params[i][0] is not None and self.model_params[i][1] is not None:
                self.model_params[i][1].zero_()
                
                
