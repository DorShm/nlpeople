import torch
from torch import nn
from torch.nn import functional as F
from General.Networks import Bilinear


class AnswerLayer(nn.Module):
    def __init__(self, config, memory_input_size, state_input_size):
        super(AnswerLayer, self).__init__()
        self.gru = getattr(nn, 'GRUCell')(memory_input_size, state_input_size)
        # self.gru = nn.GRU(memory_input_size, state_input_size)
        self.turns = int(config['turns'])
        self.predict_begin_network = Bilinear(memory_input_size, state_input_size)
        self.predict_end_network = Bilinear(memory_input_size, state_input_size)

    def forward(self, M, s_t):
        start_score_list = []
        end_score_list = []
        for turn in range(self.turns):
            # Compute P_t^begin
            # P_t_begin <=> start_vector (Softmax might better be after predict_end_network usage
            start_temp = self.predict_begin_network(M, s_t)

            # Compute P_t^end
            # P_t_end <=> end_vector
            predict_end_input = s_t + torch.bmm(start_temp.unsqueeze(1), M).squeeze(1)

            start_vector = F.softmax(start_temp, 1)
            end_vector = F.softmax(self.predict_end_network(M, predict_end_input), 1)

            # Compute β = s_t*W_5*M != s_(t+1)*W_6*M = P_t^begin
            β = start_temp
            # Compute x_(t+1) = β*M
            x_t = torch.bmm(β.unsqueeze(1), M).squeeze(1)

            # Get s_(t+1) from GRU(s_t, x_(t+1))
            s_t = self.gru(x_t, s_t)

            start_score_list.append(start_vector)
            end_score_list.append(end_vector)

        start_scores = torch.stack(start_score_list, 1)
        end_scores = torch.stack(end_score_list, 1)
        start = torch.mean(start_scores, 1)
        end = torch.mean(end_scores, 1)

        return start, end
