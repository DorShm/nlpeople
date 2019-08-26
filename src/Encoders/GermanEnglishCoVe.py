import torch
from torch import nn

class GermanEnglishCoVe(nn.Module):
  def __init__(self, config):
    super(GermanEnglishCoVe, self).__init__()
    self.input_size = int(config['input_size'])
    self.hidden_size = int(config['hidden_size'])
    self.second_input_size = int(config['second_input_size'])
    self.second_hidden_size = int(config['second_hidden_size'])
    self.dropout = float(config['dropout'])
    self.lstm_l1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=True, dropout=self.dropout)
    self.lstm_l2 = nn.LSTM(self.second_input_size, self.second_hidden_size, num_layers=1, bidirectional=True, dropout=self.dropout)

    self._load_model_weights(config['cove_model_path'])

    for param in self.parameters(): param.requires_grad = False

  def _load_model_weights(self, model_path):
    state_dict = torch.load(model_path)

    state_dict_l1 = dict([(name, param) for name, param in state_dict.items() if '0' in name])
    state_dict_l2 = dict([(name.replace('1', '0'), param) for name, param in state_dict.items() if '1' in name])

    self.lstm_l1.load_state_dict(state_dict_l1)
    self.lstm_l2.load_state_dict(state_dict_l2)

  def forward(self, x):
    output_l1, _ = self.lstm_l1(x)
    output_l2, _ = self.lstm_l2(output_l1)
    return output_l1, output_l2