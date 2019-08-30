from torch import nn


class FeedForward(nn.Module):
  def __init__(self, input_size, hidden_size, second_hidden_size):
    super(FeedForward, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.second_hidden_size = second_hidden_size
    self.fc1 = nn.Linear(self.input_size, self.hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(self.hidden_size, self.second_hidden_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    hidden = self.fc1(x)
    relu = self.relu(hidden)
    output = self.fc2(relu)
    return output


class OneLayerBRNN(nn.Module):
  def __init__(self, input_size, hidden_size, dropout=None):
    super(OneLayerBRNN, self).__init__()
    self.dropout = dropout
    self.output_size = hidden_size * 2
    self.hidden_size = hidden_size
    self.rnn = getattr(nn, self.cell_type)(input_size, hidden_size, num_layers=1, bidirectional=True)

  def forward(self, x):
    x = self.dropout(x)
    size = x.size()
    rnn_output, h = self.rnn(x)
    rnn_output = self.maxout(rnn_output, size)
    return rnn_output

  '''
  This function reduce the size of each vector by 2. If the rnn_output is of size 256 the maxout will return a vector
  of size 128
  '''
  def maxout(self, rnn_output, size):
    return rnn_output.view(size[0], self.hidden_size, 2).max(-1)[0]
