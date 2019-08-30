import torch
from torch import nn
from torchnlp.nn import Attention

class MemoryLayer:
  def __init__(self, H_Q, H_P, d):
    super(MemoryLayer, self).__init__()
    self.q_transform = nn.RNN(input_size=(2 * d), hidden_size=128, nonlinearity='relu')
    self.p_transform = nn.RNN(input_size=(2 * d), hidden_size=128, nonlinearity='relu')
    self.attention = Attention(2 * d)

  def phase_one(self, H_Q, H_P):
    H_Q_hat = self.q_transform(H_Q)
    H_P_hat = self.p_transform(H_P)
