from torch import nn
from torch.nn import functional as F


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
  def __init__(self, input_size, hidden_size):
    super(OneLayerBRNN, self).__init__()
    self.output_size = hidden_size * 2
    self.hidden_size = hidden_size
    self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, bidirectional=True)

  def forward(self, x):
    size = x.size()
    rnn_output, h = self.rnn(x)
    rnn_output = self.maxout(rnn_output, size)
    return rnn_output

  '''
  This function reduce the size of each vector by 2. If the rnn_output is of size 256 the maxout will return a vector
  of size 128
  '''
  def maxout(self, rnn_output, size):
    return rnn_output.view(size[0], size[1], self.hidden_size, 2).max(-1)[0]


class LinearSelfAttn(nn.Module):
  def __init__(self, input_size):
    super(LinearSelfAttn, self).__init__()
    self.linear = nn.Linear(input_size, 1)

  def forward(self, x):
    x_flat = x.contiguous().view(-1, x.size(-1))
    scores = self.linear(x_flat).view(x.size(0), x.size(1))
    alpha = F.softmax(scores, 1)
    return alpha.unsqueeze(1).bmm(x).squeeze(1)


class Bilinear(nn.Module):
  def __init__(self, x_size, y_size):
    super(Bilinear, self).__init__()
    self.linear = nn.Linear(y_size, x_size)

  def forward(self, x, y):
    Wy = self.linear(y)
    xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
    return xWy
