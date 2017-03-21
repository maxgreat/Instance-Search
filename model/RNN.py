
# coding: utf-8

# In[2]:

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


# In[ ]:

class RNN(nn.Module):
    """
        Define a RNN network
    """
    def __init__(self, net, hidden_size):
        super(siamese, self).__init__()
        self.features = net
        self.rnn = nn.LSTMCell(input_size=net.classifier[len(net.classifier._modules)-1], hidden_size=hidden_size)
    
    def forward(self, x, hx, cx):
        x = self.features(x)
        x = self.rnn(x, hx, cx)
        return x, hx

