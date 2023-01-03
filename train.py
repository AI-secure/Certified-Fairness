import os
import argparse
import json
import datetime
import numpy as np
import random
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import torch.nn as nn

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

from datasets import AdultDataset,CompasDataset,CrimeDataset,GermanDataset,HealthDataset,LawschoolDataset
import models
from utils import fairness_metric
from utils import Averagemeter

parser = argparse.ArgumentParser(description='Fairness Training')
parser.add_argument('-dataset', type=str, choices=['adult','compas','crime','german','health','lawschool'], default='compas')
parser.add_argument('-data_dir', type=str, default='./data/')

parser.add_argument('-protected_att', type=str)
parser.add_argument('-load', type=str)
parser.add_argument('-label', type=str)
parser.add_argument('-transfer', type=bool, default=False)
parser.add_argument('-quantiles', type=bool, default=True)

parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-batch-size', type=int, default=256)
parser.add_argument('-balanced', type=bool, default=False)

# crime: 0.05
# German: 0.05
# others: 0.001
parser.add_argument('-learning_rate', type=float, default=0.001)

parser.add_argument('-weight_decay', type=float, default=0.01)
parser.add_argument('-patience', type=float, default=5)

parser.add_argument('-num_models', type=int, default=1)

args = parser.parse_args()

# device
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init seed
def set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(SEED)

# dataset
if args.dataset=='adult':
    train_dataset = AdultDataset('train', args)
    val_dataset = AdultDataset('validation', args)
    test_dateset = AdultDataset('test', args)
elif args.dataset == 'compas':
    train_dataset = CompasDataset('train', args)
    val_dataset = CompasDataset('validation', args)
    test_dateset = CompasDataset('test', args)
elif args.dataset == 'crime':
    train_dataset = CrimeDataset('train', args)
    val_dataset = CrimeDataset('validation', args)
    test_dateset = CrimeDataset('test', args)
elif args.dataset == 'german':
    train_dataset = GermanDataset('train', args)
    val_dataset = GermanDataset('validation', args)
    test_dateset = GermanDataset('test', args)
elif args.dataset == 'health':
    train_dataset = HealthDataset('train', args)
    val_dataset = HealthDataset('validation', args)
    test_dateset = HealthDataset('test', args)
elif args.dataset == 'lawschool':
    train_dataset = LawschoolDataset('train', args)
    val_dataset = LawschoolDataset('validation', args)
    test_dateset = LawschoolDataset('test', args)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dateset, batch_size=args.batch_size, shuffle=False
)




def run(model, optimizer, loader, split, epoch):
    predictions = []
    targets = []
    protects = []
    logits_all = []
    avg_loss = Averagemeter.AverageMeter('avg_loss')
    for data, target, protected in loader:
        data, target, protected = data.to(device), target.to(device), protected.to(device)
        logits = model(data).squeeze()
        bce_loss = binary_cross_entropy(logits, target)
        if split == 'train':
            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()
        logits_all.append(logits.cpu())
        prediction = (0.5 <= nn.Sigmoid().to(device)(logits)).float().squeeze()
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
        protects.append(protected.cpu())

        avg_loss.update(bce_loss.mean().item(), len(data))

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    protects = torch.cat(protects)
    logits_all = torch.cat(logits_all)

    # if split == 'train' and epoch == args.num_epochs-1:
    #     torch.save(targets,'./logs/labels')
    #     torch.save(logits_all,'./logs/logits')
    #     torch.save(protects,'./logs/protects')

    if split == 'test':
        accuracy = accuracy_score(targets, predictions)
        print(f'Acc: {accuracy}')

        parity = fairness_metric.statistical_parity(predictions,protects)
        print(f'Parity: {parity}')

        print(f'Test loss: {avg_loss.avg}')

    return avg_loss.avg


# model preparation

# 200,20
print('length of features')
print(train_dataset.features.shape[1])
model = models.MLP([train_dataset.features.shape[1], 20, 20, 1])


# training preparation
binary_cross_entropy = nn.BCEWithLogitsLoss(
    pos_weight=train_dataset.pos_weight('train') if args.balanced else None
)
optimizer = torch.optim.Adam(
    list(model.parameters()),
    lr=args.learning_rate, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=args.patience, factor=0.5
)
# train the model
for epoch in tqdm(range(args.num_epochs)):
    model.train()
    loss_train = run(model, optimizer, train_loader, 'train', epoch)
    # print('epoch {}: lr {} loss {}'.format(epoch,optimizer.state_dict()['param_groups'][0]['lr'],loss_train))

    model.eval()
    loss_val = run(model, optimizer, val_loader, 'valid', epoch)
    scheduler.step(loss_val)

torch.save(model,f'logs/{args.dataset}_model')

# test the model
loss_test = run(model, optimizer, test_loader, 'test', 0)