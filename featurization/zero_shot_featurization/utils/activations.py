from torch import nn
from torch.nn import *
from torch.nn import functional as F

LeakyReLU
CELU
SELU
SiLU
Softmin
Softmax
Hardsigmoid
Sigmoid
Softplus
Threshold(0,1)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
