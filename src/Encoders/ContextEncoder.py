import torch
from torch import nn
from src.General.Networks import OneLayerBRNN


# TODO: create confuguration for the contextual encoder
class ContextEncoder(nn.Module):
  def __init__(self, config):
    super(ContextEncoder, self).__init__()
    self.layer_1_input_size = int(config['input_size'])
    self.layer_1_output_size = int(config['layer_1_output_size'])
    self.layer_2_input_size = self.layer_1_output_size + int(config['cove_size'])
    self.layer_2_output_size = int(config['layer_2_output_size'])
    self.dropout = float(config['dropout'])
    self.layer_1 = OneLayerBRNN(self.layer_1_input_size, self.layer_1_output_size, dropout=self.dropout)
    self.layer_2 = OneLayerBRNN(self.layer_2_input_size, self.layer_2_output_size, dropout=self.dropout)

  def forward(self, embeddings, cove_l1, cove_l2):
    layer_1_input = torch.cat((embeddings, cove_l1), 2)
    output_1 = self.layer_1(layer_1_input)
    layer_2_input = torch.cat((output_1, cove_l2), 2)
    output_2 = self.layer_2(layer_2_input)
    return torch.cat((output_1, output_2), 2)

