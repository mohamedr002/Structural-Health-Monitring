import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""
# # myhyperparameters
# x_dim = 14
# h_dim = 18
# z_dim = 12
# n_layers = 1
# n_epochs = 100
# clip = 1
# learning_rate = 1e-3
# batch_size = 10
# seed = 128
# print_every = 100
# save_every = 10
# drop=0.2
def config_vrnn(input_data,hid_dim=32,n_layers=3,drop=0.2, bid=True):
    config={"model_name":'VRNN',
    "input_dim" : input_data.size(2),
    "hid_dim" : hid_dim,
    "z_dim" : 32,
    "n_layers" : n_layers,
    "drop" : 0.2,
    "CLIP" :1,
    "fc_drop" : 0.2}
    return config
class VRNN_model(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers,drop,device):
        super().__init__()
        self.encoder = VRNN(x_dim, h_dim, z_dim, n_layers, bias=False)
        self.dropout = drop
        self.predictor = regressor((h_dim), self.dropout)
        self.device = device

    # def param_init_net(m):
    #     for name, param in m.named_parameters():
    #         nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src):
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        kld_loss, features  = self.encoder(src)
        predictions = self.predictor(features)
        return predictions, features

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())

        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias, batch_first=True)
    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).cuda()
        for t in range(x.size(1)):
            phi_x_t = self.phi_x(x[:,t])
            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # so if we want to use only for supervised learning no need for decoding
            # #decoder
            # dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            # dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(1), h)

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            # all_dec_mean.append(dec_mean_t)
            # all_dec_std.append(dec_std_t)

        return kld_loss, phi_z_t
            #    nll_loss, \
            # (all_enc_mean, all_enc_std), \
            # (all_dec_mean, all_dec_std)


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


    def _nll_gauss(self, mean, std, x):
        pass
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