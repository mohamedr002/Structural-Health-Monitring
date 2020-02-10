import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda')
import random

def config_lstm(input_data,hid_dim=32,n_layers=3,drop=0.5,bid=True):
    config={"model_name":'LSTM',
    "input_dim" : input_data.size(2),
    "hid_dim" : hid_dim,
    "out_dim" : input_data.size(2),
    "n_layers" : n_layers,
    "bid":bid,
    "drop" : 0.5,
    "CLIP" :1,
    "fc_drop" : 0.5}
    return config
class lstm(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, drop, bid, device):
        super().__init__()
        self.encoder = Encoder(input_dim, hid_dim, n_layers, drop, bid)
        self.dropout = drop
        self.predictor = regressor((hid_dim + hid_dim * bid), self.dropout)
        self.device = device

    def param_init_net(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src):
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(src)
        features=encoder_outputs[:, -1:].squeeze()
        predictions = self.predictor(features)
        return predictions, features

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout, bidirectional):
        super(Encoder,self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bid = bidirectional
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # input shape [batch_size, seq_length, input_dim]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        outputs, (hidden, cell) = self.rnn(src)
        outputs = F.dropout(torch.relu(outputs), p=0.5, training=self.training)
        # outputs are always from the top hidden layer
        return outputs, hidden, cell

class regressor(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(regressor, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, feat):
        out = self.fc1(feat)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc2(out)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        out = self.fc3(out)
        return out