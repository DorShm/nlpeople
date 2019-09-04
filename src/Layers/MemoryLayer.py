import torch
from torch import nn
from torch.nn import functional as F
from torchnlp.nn import Attention

from src.General.Networks import OneLayerBRNN


# TODO: add config to memory layer
class MemoryLayer(nn.Module):
  def __init__(self, config, d):
    super(MemoryLayer, self).__init__()
    self.q_transform = nn.RNN(input_size=(2 * d), hidden_size=128, nonlinearity='relu')
    self.p_transform = nn.RNN(input_size=(2 * d), hidden_size=128, nonlinearity='relu')
    self.attention = Attention(128, attention_type='dot')
    self.self_attention = Attention(4 * d)
    self.bi_lstm = OneLayerBRNN(input_size=(8 * d), hidden_size=128, dropout=config['dropout'])

    self.dropout = nn.Dropout(float(config['dropout']))

  def forward(self, H_Q, H_P):
    C = self.phase_one(H_Q, H_P)
    U_P = self.phase_two(H_Q, H_P, C)
    M_input = self.phase_three(U_P)
    M = self.phase_four(M_input)

    return M

  def phase_one(self, H_Q, H_P):
    out_q, H_Q_hat = self.q_transform(H_Q)
    out_h, H_P_hat = self.p_transform(H_P)

    att_output, att_weights = self.attention(H_Q_hat, H_P_hat)
    C = self.dropout(att_weights)

    return C

  def phase_two(self, H_Q, H_P, C):
    dot_product = torch.bmm(torch.transpose(H_Q, 1, 2), C)
    U_P = torch.cat((H_P, torch.transpose(dot_product, 1, 2)), 2)

    return U_P

  def phase_three(self, U_P):
    self_att_output, self_att_weights = self.self_attention(U_P, U_P)

    U_P_dropped = self.drop_diag(self_att_weights)

    U_P_transposed = torch.transpose(U_P, 1, 2)

    U_P_hat = torch.bmm(U_P_transposed, U_P_dropped)

    return torch.cat((U_P, torch.transpose(U_P_hat, 1, 2)), 2)

  def phase_four(self, memory_input):
    M = self.bi_lstm(memory_input)

    return M

  def drop_diag(self, mat):
    mask = torch.eye(mat.shape[1], mat.shape[2])

    mat[:].masked_fill_(mask.bool(), 0)

    return mat
