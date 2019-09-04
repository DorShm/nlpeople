from torch import nn
from torch.nn import functional as F


class AnswerLayer(nn.Module):
    def __init__(self):
        super(AnswerLayer, self).__init__()
        self.gru = None
        self.turns = None
        self.predict_begin_network = None
        self.predict_end_network = None
        self.softmax = F.softmax

    def forward(self, x, h_0):
        for turn in range(self.turns):
            start = self.predict_begin_network(x, h_0)
