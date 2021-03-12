import torch
import torch.nn as nn
import torch.nn.functional as F

class DatLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DatLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h_T = self.lstm(x)[0][:, -1]
        return F.softmax(self.fc(h_T), dim=1)