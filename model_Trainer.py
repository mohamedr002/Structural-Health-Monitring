import torch.nn as nn
from models.models_config import get_model_config, initlize
from pre_train_test_split import trainer
import torch
from torch.utils.data import DataLoader
from utils import *
device = torch.device('cpu')
# load data


my_data= torch.load('./data/train_test_dataset.pt')
train_dl = DataLoader(MyDataset(my_data['train_data'], my_data['train_labels']), batch_size=10, shuffle=True, drop_last=True)
test_dl = DataLoader(MyDataset(my_data['test_data'], my_data['test_labels']), batch_size=10, shuffle=False, drop_last=False)

class CNN_1D(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_1D, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=7, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.Flatten(),
            nn.Linear(32, self.hidden_dim))
        self.Classifier= nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 11))
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.Classifier(features)
        return predictions, features

model=CNN_1D(1,32,0.5).to(device)


params = {'window_length': 30, 'sequence_length': 30, 'batch_size': 10, 'input_dim': 14, 'pretrain_epoch': 40,
          'data_path': r"C:/Users/mohamedr002/OneDrive - Nanyang Technological University/PhD Codes Implementation/Deep Learning for RUL/data/processed_data/cmapps_train_test_cross_domain.pt",
          'dropout': 0.5,  'lr': 1e-4}
# load model
config = get_model_config('CNN')
# load data
trained_model=trainer(model, train_dl, test_dl,'SHM' ,config,params)