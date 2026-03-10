#https://docs.pytorch.org/docs/stable/nn.init.html
import torch
import torch.nn as nn
import math

w = torch.empty(3, 5)
nn.init.xavier_uniform_(w)

z = torch.empty(3, 5)
nn.init.kaiming_uniform_(z, mode="fan_in", nonlinearity="relu")

y = torch.empty(3, 5)
nn.init.kaiming_uniform_(y, a=0, nonlinearity="leaky_relu")
