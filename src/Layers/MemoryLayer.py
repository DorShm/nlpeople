import torch
from torch import nn
from torchnlp.nn import Attention
from src.General.Networks import OneLayerBRNN


class MemoryLayer(nn.Module):
  def __init__(self, config, d, cuda_on):
    super(MemoryLayer, self).__init__()
    self.cuda_on = cuda_on
    self.q_transform = nn.RNN(input_size=(int(config['transform_input_size']) * d),
                              hidden_size=int(config['hidden_size']), nonlinearity=config['nonlinearity'])

    self.p_transform = nn.RNN(input_size=(int(config['transform_input_size']) * d),
                              hidden_size=int(config['hidden_size']), nonlinearity=config['nonlinearity'])

    self.attention = Attention(int(config['hidden_size']), attention_type='dot')
    self.self_attention = Attention(int(config['attention_input_size']) * d)
    self.bi_lstm = OneLayerBRNN(input_size=(int(config['bi_lstm_input_size']) * d),
                                hidden_size=int(config['hidden_size']))

    self.output_size = int(config['hidden_size'])
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
    if self.cuda_on:
      mask = torch.eye(mat.shape[1], mat.shape[2]).cuda()
      mat_dropped = mat.clone()[:].masked_fill_(mask.bool(), 0)
    else:
      mask = torch.eye(mat.shape[1], mat.shape[2])
      mat_dropped = mat.clone()[:].masked_fill_(mask.bool(), 0)

    return mat_dropped
