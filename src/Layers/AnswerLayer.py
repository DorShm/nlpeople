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
        # TODO: change name from projection
        self.projection = nn.Linear()

    def forward(self, M, s_t):
        start_score_list = []
        end_score_list = []
        for turn in range(self.turns):
            # Compute P_t^begin
            # P_t_begin <=> start_vector (Softmax might better be after predict_end_network usage
            start_vector = F.softmax(self.predict_begin_network(M, s_t), 1)

            # Compute P_t^end
            # P_t_end <=> end_vector
            # TODO: Change this shitty name "square braces"
            square_braces = s_t + torch.bmm(start_vector, M)
            end_vector = F.softmax(self.predict_end_network(M, square_braces), 1)

            # Compute β = s_t*W_5*M != s_(t+1)*W_6*M = P_t^begin
            β = start_vector
            # Compute x_(t+1) = β*M
            x_t = torch.bmm(M, β)

            # Get s_(t+1) from GRU(s_t, x_(t+1))
            s_t = self.gru(x_t, s_t)

            start_score_list.append(start_vector)
            end_score_list.append(end_vector)

        start = torch.mean(start_score_list)
        end = torch.mean(end_score_list)

        return start, end
