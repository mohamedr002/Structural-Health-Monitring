import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda')
import random

class cnn_fe(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(cnn_fe, self).__init__()
        self.out_dim=out_dim
        self.input_dim=input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1))
        self.fc1 = nn.Linear(8, self.out_dim)  # changed from

    def forward(self, input):
        conv_out = self.encoder(input)
        conv_out = F.dropout(conv_out, p=0.5)  # we didn't need it when source domain is zero condition
        feat = self.fc1(conv_out.view(conv_out.shape[0], -1))
        return feat
class cnn_pred(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(cnn_pred, self).__init__()
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

class CNN(nn.Module):
    def __init__(self, in_dim,out_dim, dropout, device):
        super().__init__()
        self.encoder = cnn_fe(in_dim, out_dim)
        self.dropout = dropout
        self.predictor = cnn_pred(self.encoder.out_dim, self.dropout)
        self.device = device
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.predictor(features)
        return predictions, features
def config_cnn(input_data):
    cnn_config={"model_name":"CNN",
    "input_dim" : input_data.size(2),
    "CLIP" :1,
    "out_dim" : 32,
    "fc_drop" : 0.5}
    return cnn_config