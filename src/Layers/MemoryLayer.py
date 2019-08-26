import torch
from torch import nn

class MemoryLayer(nn.Module):
  def __init__(self):
    super(MemoryLayer, self).__init__()

  def forward(self, *input, **kwargs):
    pass