import torch
import torch.nn as nn
import torch.nn.functional as F

class DatLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(DatLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * seq_length, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x