import torch
from torch import nn

#TODO: Change Encoder to use 2 layer COV
class ContextEncoder(nn.Module):
  def __init__(self, config):
    super(ContextEncoder, self).__init__()
    self.input_size = int(config['input_size'])
    self.hidden_size = int(config['hidden_size'])
    self.dropout = float(config['dropout'])
    self.lstm_l1 = nn.LSTM(self.l1_input_size, self.l1_hidden_size, num_layers=1, bidirectional=True, dropout=self.dropout)
    self.lstm_l2 = nn.LSTM(self.l2_input_size, self.l2_hidden_size, num_layers=1, bidirectional=True, dropout=self.dropout)

    self._load_model_weights(config['cove_model_path'])

    for param in self.parameters(): param.requires_grad = False

  def _load_model_weights(self, model_path):
    state_dict = torch.load(model_path)

    state_dict_l1 = [(name, param) for name, param in state_dict.items() if '0' in name]
    state_dict_l2 = [(name, param) for name, param in state_dict.items() if '1' in name]

    self.lstm_l1.load_state_dict(state_dict_l1)
    self.lstm_l2.load_state_dict(state_dict_l2)

  def forward(self, x):
    output_l1, _ = self.lstm_l1(x)
    output_l2, _ = self.lstm_l2(output_l1)
    return output_l1, output_l2