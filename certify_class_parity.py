import argparse
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from sklearn.metrics import (accuracy_score)

from datasets import AdultDataset
from utils import fairness_metric
from utils import Averagemeter
import distribution_shift.label_shifting as label_shifting
import plot.plot as plot

parser = argparse.ArgumentParser(description='Fairness Certificate')
parser.add_argument('-dataset', type=str, choices=['adult','chexpert'], default='adult')
parser.add_argument('-protected_att', type=str, default='sex')

parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-batch-size', type=int, default=256)
parser.add_argument('-balanced', type=bool, default=False)

parser.add_argument('-num_generated_distributions', type=int, default=5000)
parser.add_argument('-path_sampled_distances', type=str, default='./logs/Hellinger_Distance_Q_adult')
parser.add_argument('-path_sampled_loss', type=str, default='./logs/Losses_Q_adult')
parser.add_argument('-path_figure', type=str, default='./data/fig_adult')

parser.add_argument('-num_models', type=int, default=1)

args = parser.parse_args()


# device
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init seed
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)

if args.dataset=='adult':
    train_dataset = AdultDataset('test', args)
    loader_P = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    def run(model, loader):
        predictions = []
        targets = []
        protects = []
        # logits_all = []
        loss_all = []
        avg_loss = Averagemeter.AverageMeter('avg_loss')
        for data, target, protected in loader:
            data, target, protected = data.to(device), target.to(device), protected.to(device)
            logits = model(data)
            bce_loss = binary_cross_entropy(logits, target.unsqueeze_(-1))
            # logits_all.append(logits.cpu())
            prediction = (0.5 <= nn.Sigmoid().to(device)(logits)).float().squeeze()
            predictions.append(prediction.detach().cpu())
            targets.append(target.detach().cpu())
            protects.append(protected.cpu())

            avg_loss.update(bce_loss.mean().item(), len(data))

            loss_no_reduct = binary_cross_entropy_no_reduction(logits, target).squeeze_(-1)
            loss_all.append(loss_no_reduct)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        accuracy = accuracy_score(targets, predictions)
        print(f'Acc of distribution P: {accuracy}')
        protects = torch.cat(protects)
        # logits_all = torch.cat(logits_all)
        loss_all = torch.cat(loss_all)

        return avg_loss.avg, loss_all, predictions, protects, targets

    loss_Ps = []
    distances = []
    loss_Qs = []
    hellinger_distances = None
    upper_bounds = None
    for i in range(args.num_models):
        print(f'Certifying model #{i}')

        model = torch.load(f'./logs/tmp_model')
        binary_cross_entropy = nn.BCEWithLogitsLoss()
        binary_cross_entropy_no_reduction = nn.BCEWithLogitsLoss(
            reduction='none'
        )

        loss_P, loss_all, predictions, protects, targets = run(model,loader_P)
        print('loss of distribution P: {}'.format(loss_P))
        parity = fairness_metric.statistical_parity(predictions, protects)
        print(f'Demographic parity of distribution P: {parity}')
        targets = targets.squeeze(-1)
        class_parity = abs(torch.mean(loss_all[targets==0]).item()-torch.mean(loss_all[targets==1]).item())
        print(f'Class parity of distribution P: {class_parity}')
        dists = []
        losses = []
        fairness = []
        group_00, group_01, group_10, group_11 = [], [], [], []
        for i in range(len(train_dataset.labels)):
            if train_dataset.protected[i] == False and train_dataset.labels[i] == 0:
                group_00.append(i)
            elif train_dataset.protected[i] == False and train_dataset.labels[i] == 1:
                group_01.append(i)
            elif train_dataset.protected[i] == True and train_dataset.labels[i] == 0:
                group_10.append(i)
            elif train_dataset.protected[i] == True and train_dataset.labels[i] == 1:
                group_11.append(i)

        print(f'The base rate gap of distribution P: {abs(1.0*len(group_01)/(len(group_00)+len(group_01)) - 1.0*len(group_11)/(len(group_10)+len(group_11)))}')
        print(f'The number of instances in distribution P: {len(group_00)+len(group_01)+len(group_10)+len(group_11)}')

        seeds = list(range(args.num_generated_distributions))
        for i in tqdm(range(args.num_generated_distributions)):
            # generate shifted distribution
            indices_Q_0, indices_Q_1, dist = label_shifting.label_shifting_binary_label_binary_attr_class_parity(seeds[i], train_dataset,group_00,group_01,group_10,group_11)
            if len(indices_Q_0+indices_Q_1)<5000:
                continue
            loss_Q = abs(torch.mean(loss_all[indices_Q_0]).item() -  torch.mean(loss_all[indices_Q_1]).item())
            dists.append(dist)
            losses.append(loss_Q)

        distances.append(dists)
        loss_Qs.append(losses)
    # print(hellinger_distances)
    # print(upper_bounds)
    plot.make_plot(None, None, None, distances, loss_Qs, class_parity, args.path_figure)
