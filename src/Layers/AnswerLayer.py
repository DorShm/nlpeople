import torch
from torch import nn
from torch.nn import functional as F


class AnswerLayer(nn.Module):
    def __init__(self, turns):
        super(AnswerLayer, self).__init__()
        self.gru = nn.GRU()
        self.turns = turns
        self.predict_begin_network = nn.Bilinear()
        self.predict_end_network = nn.Bilinear()
        self.projection = nn.Linear()

    def forward(self, M, s_t):
        start_score_list = []
        end_score_list = []
        for turn in range(self.turns):
            start_vector = self.predict_begin_network(M, s_t)
            end_vector = self.predict_end_network(M, s_t + start_vector)
            x_t = F.softmax(torch.bmm(start_vector, M))
            x_t = torch.bmm(x_t, M)
            s_t = self.gru(x_t, s_t)
            start_score_list.append(start_vector)
            end_score_list.append(end_vector)
