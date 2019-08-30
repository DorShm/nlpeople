import torch
from torch import nn
from torch.nn import functional as F
from torchnlp.nn import Attention

from src.General.Networks import OneLayerBRNN


class MemoryLayer:
  def __init__(self, H_Q, H_P, d, dropout=0.4):
    super(MemoryLayer, self).__init__()
    self.q_transform = nn.RNN(input_size=(2 * d), hidden_size=128, nonlinearity='relu')
    self.p_transform = nn.RNN(input_size=(2 * d), hidden_size=128, nonlinearity='relu')
    self.attention = Attention(2 * d)
    self.self_attention = Attention(4 * d)
    self.bi_lstm = OneLayerBRNN()

    self.dropout = nn.Dropout(dropout)

  def run(self, H_Q, H_P):
    C = self.phase_one(H_Q, H_P)
    U_P = self.phase_two(H_Q, H_P, C)
    M_input = self.phase_three(U_P)
    M = self.phase_four(M_input)

    return M

  def phase_one(self, H_Q, H_P):
    H_Q_hat = self.q_transform(H_Q)
    H_P_hat = self.p_transform(H_P)

    # TODO: Apply dropout
    att_output, _ = self.attention(H_Q_hat, H_P_hat)
    C = self.dropout(att_output)

    return C

  def phase_two(self, H_Q, H_P, C):
    U_P = torch.cat(H_P, torch.dot(H_Q, C))

    return U_P

  def phase_three(self, U_P):
    self_att_output = self.self_attention(U_P, U_P)
    # TODO: Apply drop_diag (whatever it means...)
    U_P_dropped = U_P

    U_P_hat = torch.dot(U_P, U_P_dropped)

    return torch.cat(U_P, U_P_hat)

  def phase_four(self, memory_input):
    M = self.bi_lstm(memory_input)

    return M
