##############################################################################
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./trained_models/
# All runs and experiment
# The file ./opts.py stores the options
# The file ./train_eval.py stores the training and test strategy
# The file ./main.py should be simple
#################################################################################
import warnings
import torch
from torch import optim
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from train_eval import train, evaluate

# fix_randomness(5)
import torch.nn as nn
from sklearn.metrics import classification_report
device = torch.device('cuda')


# 30 epochs for all, while 5 epochs for FD003---> pretraining number of epochs
def trainer(model, train_dl, test_dl, data_id, config, params):
    # criteierion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    target_names = ['Healthy','D1','D2','D3','D4','D5','D6','D7', 'D8','D9','D10']
    for epoch in range(params['pretrain_epoch']):
        start_time = time.time()
        train_loss, train_pred, train_labels = train(model, train_dl, optimizer, criterion, config)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        # Evaluate on the test set
        test_loss,_, _= evaluate(model, test_dl, criterion, config)
        print('=' * 89)
        print(f'\t  Performance on test set::: Loss: {test_loss:.3f} ')#| Score: {test_score:7.3f}')
        train_labels = torch.stack(train_labels).view(-1)
        train_pred = torch.stack(train_pred).view(-1)
        print(classification_report(train_labels, train_pred, target_names=target_names))
    # saving last epoch model
    # checkpoint1 = {'model': model,
    #                'epoch': epoch,
    #                'state_dict': model.state_dict(),
    #                'optimizer': optimizer.state_dict()}
    # torch.save(checkpoint1,
    #            f'./checkpoints/{config["model_name"]}/pretrained_{config["model_name"]}_{data_id}_tuned.pt')

    # Evaluate on the test set
    test_loss, y_pred, y_true = evaluate(model, test_dl, criterion, config)
    print('=' * 89)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} ')#| Score: {test_score:7.3f}')
    print('=' * 89)
    y_true = torch.stack(y_true).view(-1)
    y_pred = torch.stack(y_pred).view(-1)
    print(classification_report(y_true, y_pred, target_names=target_names))

    print('| End of Pre-training  |')
    print('=' * 89)
    return model


