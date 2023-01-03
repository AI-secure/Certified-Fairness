import argparse
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import torch.nn as nn

from sklearn.metrics import (accuracy_score, f1_score)
from tqdm import tqdm

from datasets import AdultDataset,CompasDataset,CrimeDataset,GermanDataset,HealthDataset,LawschoolDataset
import models
from utils import fairness_metric
from utils import Averagemeter

from fairness_bound_label_shifting_finite_sampling import fairness_certification_bound_label_shifting_finite_sampling
from fairness_bound_general_shifting_finite_sampling import fairness_upper_bound_general_shifting_finite_sampling

parser = argparse.ArgumentParser(description='Fairness Training')
parser.add_argument('-dataset', type=str,
                    choices=['adult','compas','crime','german','health','lawschool'], default='compas')

parser.add_argument('-protected_att', type=str)
parser.add_argument('-load', type=str)
parser.add_argument('-label', type=str)
parser.add_argument('-transfer', type=bool, default=False)
parser.add_argument('-quantiles', type=bool, default=True)

parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-batch-size', type=int, default=256)
parser.add_argument('-balanced', type=bool, default=True)


# crime: 0.05
# German: 0.05
# others: 0.001
parser.add_argument('-learning_rate', type=float, default=0.001)

parser.add_argument('-weight_decay', type=float, default=0.01)
parser.add_argument('-patience', type=float, default=5)

parser.add_argument('-num_models', type=int, default=1)
parser.add_argument('-num_generated_distributions', type=int, default=0)

parser.add_argument('-path_figure', type=str, default='./data/fig_german_general_shifting_loss')

parser.add_argument('-pretrain', type=bool, default=False)

parser.add_argument('-num_points_bound', type=int, default=3)
parser.add_argument('-certificate_low', type=float, default=0.1)
parser.add_argument('-certificate_high', type=float, default=0.6)
parser.add_argument('-gamma_k', type=float, default=0.0)
parser.add_argument('-gamma_r', type=float, default=0.0)
parser.add_argument('-interval_kr', type=float, default=0.005) # if gamma is very large (near 0.5), interval_kr=0.002 (in the ablation study)

parser.add_argument('-solve_label_shifting_finite_sampling', type=bool, default=False)
parser.add_argument('-solve_general_shifting_finite_sampling', type=bool, default=True)
parser.add_argument('-solve_grammian', type=bool, default=False)

parser.add_argument('-M', type=float, default=1.0)
parser.add_argument('-loss_function', type=str, default='JSD')
parser.add_argument('-use_loss', type=int, default=0)
parser.add_argument('-confidence', type=float, default=0.1)

parser.add_argument('-finite_sampling', type=bool, default=False)


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
    train_dataset = AdultDataset('test', args)
    test_dataset = AdultDataset('test', args, change_covariate=True)
elif args.dataset == 'compas':
    train_dataset = CompasDataset('test', args, test_size=0.9)
    test_dataset = CompasDataset('test', args, test_size=0.9, change_covariate=True)
elif args.dataset == 'crime':
    train_dataset = CrimeDataset('test', args, test_size=0.9)
    test_dataset = CrimeDataset('test', args, test_size=0.9, change_covariate=True)
elif args.dataset == 'german':
    train_dataset = GermanDataset('test', args, test_size=0.9)
    test_dataset = GermanDataset('test', args, test_size=0.9, change_covariate=True)
elif args.dataset == 'health':
    train_dataset = HealthDataset('test', args)
    test_dataset = HealthDataset('test', args, change_covariate=True)
elif args.dataset == 'lawschool':
    train_dataset = LawschoolDataset('test', args)
    test_dataset = LawschoolDataset('test', args, change_covariate=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False
)

import torch.nn.functional as F


def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log2() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, dim=1)
    return output

class JSDLoss(torch.nn.Module):
    def __init__(self, num_classes, reduce='mean'):
        super(JSDLoss, self).__init__()
        self.num_classes = num_classes
        self.reduce = reduce

    def set_reduce(self, reduce):
        self.reduce = reduce

    def forward(self, predictions, labels):
        preds = F.softmax(predictions, dim=1)
        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels, preds]
        mean_distrib = sum(distribs) / len(distribs)
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log2()

        kldivs1 = custom_kl_div(mean_distrib_log, labels)
        kldivs2 = custom_kl_div(mean_distrib_log, preds)

        if self.reduce == 'mean':
            return 0.5 * (kldivs1.mean() + kldivs2.mean())
        if self.reduce == 'none':
            return 0.5 * (kldivs1 + kldivs2)
        if self.reduce == 'sum':
            return 0.5 * (kldivs1.sum() + kldivs2.sum())

        assert False

def run(model, optimizer, loader, split, epoch):
    predictions = []
    targets = []
    protects = []
    logits_all = []
    avg_loss = Averagemeter.AverageMeter('avg_loss')
    for data, target, protected in loader:
        data, target, protected = data.to(device), target.to(device), protected.to(device)
        logits = model(data)
        bce_loss = binary_cross_entropy(logits, target.unsqueeze_(-1))
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

    accuracy = accuracy_score(targets, predictions)

    parity = fairness_metric.statistical_parity(predictions,protects)

    print(f'Epoch: {epoch}  Loss: {avg_loss.avg}  Acc: {accuracy}  Parity: {parity}')

    return avg_loss.avg


# model preparation
model = models.MLP([103, 20, 20, 1])

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
binary_cross_entropy_no_reduction = nn.BCEWithLogitsLoss(
    reduction='none'
)
def evaluate(model, loader):
    predictions = []
    targets = []
    protects = []
    loss_all = []
    logits_all = []
    avg_loss = Averagemeter.AverageMeter('avg_loss')
    for data, target, protected in loader:
        data, target, protected = data.to(device), target.to(device), protected.to(device)
        logits = model(data)

        bce_loss = binary_cross_entropy(logits, target.unsqueeze_(-1))
        loss_no_reduct = binary_cross_entropy_no_reduction(logits, target).squeeze_(-1)
        loss_all.append(loss_no_reduct)

        logits_all.append(logits.cpu())
        prediction = (0.5 <= nn.Sigmoid().to(device)(logits)).float().squeeze().detach().cpu()
        target = target.squeeze().detach().cpu()

        predictions.append(prediction)
        targets.append(target)
        protects.append(protected.cpu())

        avg_loss.update(bce_loss.mean().item(), len(data))


    predictions = torch.cat(predictions)

    targets = torch.cat(targets)
    accuracy = accuracy_score(targets, predictions)
    print(f'Acc of distribution P: {accuracy}')
    protects = torch.cat(protects)
    logits_all = torch.cat(logits_all)
    loss_all = torch.cat(loss_all)


    classification_acc = (predictions==targets).float()
    # print(classification_acc)

    print(f'Acc of distribution P: {torch.mean(classification_acc).item()}')

    # np.average(classification_acc)

    return avg_loss.avg, loss_all, logits_all, targets, classification_acc


# if args.pretrain:
#     # train the model
#     for epoch in tqdm(range(args.num_epochs)):
#         model.train()
#         loss_train = run(model, optimizer, train_loader, 'train', epoch)
#         # print('epoch {}: lr {} loss {}'.format(epoch,optimizer.state_dict()['param_groups'][0]['lr'],loss_train))
#
#         model.eval()
#         loss_val = run(model, optimizer, val_loader, 'valid', epoch)
#         scheduler.step(loss_val)
#
#     torch.save(model,f'logs/{args.dataset}_model_general_shifting')

model = torch.load(f'logs/{args.dataset}_model')



# test the model
_, _, logits_P, tar_P, classification_acc_P = evaluate(model, train_loader)
_, _, logits_Q, tar_Q, classification_acc_Q = evaluate(model, test_loader)


criterion = JSDLoss(num_classes=2, reduce='mean')
loss_all_P, loss_all_Q = [], []

for i in range(len(logits_P)):
    x = torch.tensor([[1.0-logits_P[i],logits_P[i]]])
    y = tar_P[i:i+1].to(torch.int64)
    loss_all_P.append(criterion(x, y))
loss_all_P = torch.tensor(loss_all_P)

for i in range(len(logits_Q)):
    x = torch.tensor([[1.0-logits_Q[i],logits_Q[i]]])
    y = tar_Q[i:i+1].to(torch.int64)
    loss_all_Q.append(criterion(x, y))
loss_all_Q = torch.tensor(loss_all_Q)

losses_all = []
print(f'loss_Q: {torch.mean(loss_all_Q).item()}')

# print()
# print(len(classification_acc_P))
# print(torch.mean(classification_acc_P).item())

import math

group_00_P, group_01_P, group_10_P, group_11_P = [], [], [], []
for i in range(len(train_dataset.labels)):
    if train_dataset.protected[i] == False and train_dataset.labels[i] == 0:
        group_00_P.append(i)
    elif train_dataset.protected[i] == False and train_dataset.labels[i] == 1:
        group_01_P.append(i)
    elif train_dataset.protected[i] == True and train_dataset.labels[i] == 0:
        group_10_P.append(i)
    elif train_dataset.protected[i] == True and train_dataset.labels[i] == 1:
        group_11_P.append(i)
group_P = [group_00_P, group_01_P, group_10_P, group_11_P]
N_P = len(group_00_P) + len(group_01_P) + len(group_10_P) + len(group_11_P)
Num_P = [len(group_00_P), len(group_01_P), len(group_10_P), len(group_11_P)]
lambda_P = [1.0*len(group_00_P)/N_P, 1.0*len(group_01_P)/N_P, 1.0*len(group_10_P)/N_P, 1.0*len(group_11_P)/N_P]

group_00_Q, group_01_Q, group_10_Q, group_11_Q = [], [], [], []
for i in range(len(test_dataset.labels)):
    if test_dataset.protected[i] == False and test_dataset.labels[i] == 0:
        group_00_Q.append(i)
    elif test_dataset.protected[i] == False and test_dataset.labels[i] == 1:
        group_01_Q.append(i)
    elif test_dataset.protected[i] == True and test_dataset.labels[i] == 0:
        group_10_Q.append(i)
    elif test_dataset.protected[i] == True and test_dataset.labels[i] == 1:
        group_11_Q.append(i)
group_Q = [group_00_Q, group_01_Q, group_10_Q, group_11_Q]
N_Q = len(group_00_Q) + len(group_01_Q) + len(group_10_Q) + len(group_11_Q)
Num_Q = [len(group_00_Q), len(group_01_Q), len(group_10_Q), len(group_11_Q)]
lambda_Q = [1.0*len(group_00_Q)/N_Q, 1.0*len(group_01_Q)/N_Q, 1.0*len(group_10_Q)/N_Q, 1.0*len(group_11_Q)/N_Q]


def get_label_distribution_Q(lamb_Q, lamb_P, count_P):
    pivot = np.argmax(np.array(lamb_Q) / np.array(lamb_P))
    count_Q = [0] * 8

    # if random.uniform(0,1)<0.7:
    #     count_Q[pivot] = random.randint(0,count_P[pivot])
    # else:
    #     count_Q[pivot] = random.randint(0, count_P[pivot]//10)
    # count_Q[pivot] = random.randint(0, count_P[pivot])
    count_Q[pivot] = count_P[pivot]
        # print(lamb_P)
    # print(count_P)
    # print(count_Q)
    for i in range(8):
        count_Q[i] = int((1.0 * lamb_Q[i] / lamb_Q[pivot]) * count_Q[pivot])
    return count_Q

def group_sample(group, N_):
    indices = []
    for i in range(4):
        # if N_[i]>len(group[i]):
        #     print('error')
        #     print(N_[i])
        #     print(len(group[i]))
        indices = indices + list(np.random.choice(group[i], N_[i], replace=False))
    return indices

distances = []
classification_errors = []


# lambda_P = lambda_P[::-1]
# lambda_Q = lambda_Q[::-1]
print(f'lambda_P: {lambda_P}')
print(f'lambda_Q: {lambda_Q}')

import distribution_shift.label_shifting as label_shifting
seeds = list(range(args.num_generated_distributions))
# for i in tqdm(range(args.num_generated_distributions)):
    # # generate shifted distribution
    # indices_Q, dist = label_shifting.label_shifting_binary_label_binary_attr(seeds[i], test_dataset,group_00_P,group_01_P,group_10_P,group_11_P,args.gamma_k, args.gamma_r)
    # # if len(indices_Q)<5000:
    # #     continue
    # # if dist > args.certificate_high - 0.02:
    # #     continue
    #
    # acc_Q = 1-torch.mean(classification_acc_P[indices_Q]).item()
    # distances.append(dist)
    # classification_errors.append(acc_Q)
seeds = list(range(args.num_generated_distributions))
for i in tqdm(range(args.num_generated_distributions)):
    random.seed(seeds[i])
    k = random.uniform(args.gamma_k,1-args.gamma_k)
    r = random.uniform(args.gamma_r,1-args.gamma_r)

    tmp = random.uniform(0,1)
    if tmp > 0.7:
        KK = 0.99
        RR=1.0
    elif tmp>0.3:
        KK=0.7 # KK = 0.5 or 0.4  (for adult,compas,lawschool,health) new params are for crime and german
        RR=0.9 # RR = 0.7 (for adult,compas,lawschool,health)
    else:
        KK=0.2 # KK=0.0 (for adult,compas,lawschool,health)
        RR=0.4 # RR =0.2(for adult,compas,lawschool,health)

    alpha_00_P = random.uniform(k * r / lambda_P[0] * KK, k * r / lambda_P[0] * RR)
    alpha_01_P = random.uniform(k * (1 - r) / lambda_P[1] * KK,k * (1 - r) / lambda_P[1] * RR)
    alpha_10_P = random.uniform((1 - k) * r / lambda_P[2] * KK,(1 - k) * r / lambda_P[2] * RR)
    alpha_11_P = random.uniform((1 - k) * (1 - r) / lambda_P[3] * KK,(1 - k) * (1 - r) / lambda_P[3] * RR)

    # alpha_00_P = k * r / lambda_P[0]
    # alpha_01_P = k * (1-r) / lambda_P[1]
    # alpha_10_P = (1-k) * r / lambda_P[2]
    # alpha_11_P = (1-k) * (1-r) / lambda_P[3]


    # alpha_00_P = random.uniform(0, 10)
    # alpha_01_P = random.uniform(0, 10)
    # alpha_10_P = random.uniform(0, 10)
    # alpha_11_P = random.uniform(0, 10)

    # alpha_00_Q = random.uniform(0, 1)
    # alpha_01_Q = random.uniform(0, 1)
    # alpha_10_Q = random.uniform(0, 1)
    # alpha_11_Q = random.uniform(0, 1)

    # alpha_00_Q = 1 - alpha_00_P
    # alpha_01_Q = 1 - alpha_01_P
    # alpha_10_Q = 1 - alpha_10_P
    # alpha_11_Q = 1 - alpha_11_P

    # x = alpha_00_P*lambda_P[0] + alpha_00_Q * lambda_Q[0]
    # y = alpha_01_P*lambda_P[1] + alpha_01_Q * lambda_Q[1]
    # r = x / (x + y)
    # z = alpha_10_P*lambda_P[2] + alpha_10_Q * lambda_Q[2]
    # k = x / (x + z)
    # alpha_11_P = ((1-k)*(1-r)-lambda_Q[3]) / (lambda_P[3] - lambda_Q[3])
    # alpha_11_Q = 1 - alpha_11_P
    alpha_00_Q = (k * r - alpha_00_P*lambda_P[0]) / lambda_Q[0]
    alpha_01_Q = (k * (1-r) - alpha_01_P * lambda_P[1]) / lambda_Q[1]
    alpha_10_Q = ((1-k) * r - alpha_10_P * lambda_P[2]) / lambda_Q[2]
    alpha_11_Q = ((1-k) * (1-r) - alpha_11_P * lambda_P[3]) / lambda_Q[3]

    # print(f'{k} {r} {alpha_00} {alpha_01} {alpha_10} {alpha_11}')

    # if alpha_00_Q < 1e-10:
    #     continue
    # if alpha_01_Q < 1e-10:
    #     continue
    # if alpha_10_Q < 1e-10:
    #     continue
    # if alpha_11_P < 1e-10:
    #     continue
    # if alpha_11_Q < 1e-10:
    #     continue

    dist = math.sqrt(0.5*(1+alpha_00_P*lambda_P[0]+alpha_01_P*lambda_P[1]+alpha_10_P*lambda_P[2]+alpha_11_P*lambda_P[3]
                          +alpha_00_Q*lambda_Q[0]+alpha_01_Q*lambda_Q[1]+alpha_10_Q*lambda_Q[2]+alpha_11_Q*lambda_Q[3])
                     -(math.sqrt(alpha_00_P)*lambda_P[0]+math.sqrt(alpha_01_P)*lambda_P[1]+math.sqrt(alpha_10_P)*lambda_P[2]+math.sqrt(alpha_11_P)*lambda_P[3]))
    # C = math.sqrt(lambda_P[0]*(alpha_00_P*lambda_P[0]+alpha_00_Q*lambda_Q[0]))*math.sqrt(alpha_00_P)\
    #     +math.sqrt(lambda_P[1]*(alpha_01_P*lambda_P[1]+alpha_01_Q*lambda_Q[1]))*math.sqrt(alpha_01_P)\
    #     +math.sqrt(lambda_P[2]*(alpha_10_P*lambda_P[2]+alpha_10_Q*lambda_Q[2]))*math.sqrt(alpha_10_P)\
    #     +math.sqrt(lambda_P[3]*(alpha_11_P*lambda_P[3]+alpha_11_Q*lambda_Q[3]))*math.sqrt(alpha_11_P)
    # if C > 1- 1e-10:
    #     continue
    # dist = math.sqrt(1-C)
    if dist > args.certificate_high:
        continue

    lamb_final_P = [alpha_00_P * lambda_P[0], alpha_01_P * lambda_P[1], alpha_10_P* lambda_P[2], alpha_11_P * lambda_P[3]]
    lamb_final_Q = [alpha_00_Q * lambda_Q[0], alpha_01_Q * lambda_Q[1], alpha_10_Q* lambda_Q[2], alpha_11_Q * lambda_Q[3]]


    count_ = get_label_distribution_Q(lamb_final_P+lamb_final_Q, lambda_P+lambda_Q, Num_P+Num_Q)

    indices_P = []
    group_00_shuffle_P = list(group_00_P)
    random.shuffle(group_00_shuffle_P)
    indices_P = indices_P + group_00_shuffle_P[:count_[0]]
    group_01_shuffle_P = list(group_01_P)
    random.shuffle(group_01_shuffle_P)
    indices_P = indices_P + group_01_shuffle_P[:count_[1]]
    group_10_shuffle_P = list(group_10_P)
    random.shuffle(group_10_shuffle_P)
    indices_P = indices_P + group_10_shuffle_P[:count_[2]]
    group_11_shuffle_P = list(group_11_P)
    random.shuffle(group_11_shuffle_P)
    indices_P = indices_P + group_11_shuffle_P[:count_[3]]

    indices_Q = []
    group_00_shuffle_Q = list(group_00_Q)
    random.shuffle(group_00_shuffle_Q)
    indices_Q = indices_Q + group_00_shuffle_Q[:count_[4]]
    group_01_shuffle_Q = list(group_01_Q)
    random.shuffle(group_01_shuffle_Q)
    indices_Q = indices_Q + group_01_shuffle_Q[:count_[5]]
    group_10_shuffle_Q = list(group_10_Q)
    random.shuffle(group_10_shuffle_Q)
    indices_Q = indices_Q + group_10_shuffle_Q[:count_[6]]
    group_11_shuffle_Q = list(group_11_Q)
    random.shuffle(group_11_shuffle_Q)
    indices_Q = indices_Q + group_11_shuffle_Q[:count_[7]]

    # indices_P, dist = label_shifting.label_shifting_binary_label_binary_attr(seeds[i], test_dataset, group_00_P, group_01_P,
    #                                                                          group_10_P, group_11_P, args.gamma_k,
    #                                                                          args.gamma_r)
    #
    # print(len(indices_P))


    distances.append(dist)
    # c_1 = alpha_00_P * lambda_P[0] + alpha_01_P * lambda_P[1] + alpha_10_P* lambda_P[2] + alpha_11_P * lambda_P[3]
    # c_2 = alpha_00_Q * lambda_Q[0] + alpha_01_Q * lambda_Q[1] + alpha_10_Q* lambda_Q[2] + alpha_11_Q * lambda_Q[3]

    # in the worst case, the classification accuracy of distribution Q is 0 (because distribution P and distribution Q are disjoint)
    # c_2 = 0

    c_1 = 1.0 * sum(count_[:4]) / sum(count_)
    c_2 = 1.0 * sum(count_[4:]) / sum(count_)
    # print(f'{c_1} {c_2} {dist}')

    classification_errors.append(1-c_1*(torch.mean(classification_acc_P[indices_P]).item()) - c_2 * (torch.mean(classification_acc_Q[indices_Q]).item()) )

    losses_all.append(c_1*torch.mean(loss_all_P[indices_P]).item()+c_2*torch.mean(loss_all_Q[indices_Q]).item())

    # classification_errors.append(1 - (torch.mean(classification_acc_P[indices_P]).item()))

    #
    # lamb_ori = [lambda_P[0],lambda_P[1],lambda_P[2],lambda_P[3],lambda_Q[0],lambda_Q[1],lambda_Q[2],lambda_Q[3]]

    #
    # # DEBUG
    # a = 1.0
    # b = 0.0
    # for i in range(4):
    #     lamb_ori[i] = lamb_ori[i] * a
    # for i in range(4,8):
    #     lamb_ori[i] = lamb_ori[i] * b
    #
    # count_ori = [Num_P[0],Num_P[1],Num_P[2],Num_P[3],Num_Q[0],Num_Q[1],Num_Q[2],Num_Q[3]]
    # lamb_final = [alpha_00_P * lambda_P[0], alpha_01_P * lambda_P[1], alpha_10_P* lambda_P[2], alpha_11_P * lambda_P[3],
    #               alpha_00_Q * lambda_Q[0], alpha_01_Q * lambda_Q[1], alpha_10_Q* lambda_Q[2], alpha_11_Q * lambda_Q[3]]
    #
    # lamb_ori = lambda_P
    # count_ori = Num_P
    # lamb_final = [alpha_00_P * lambda_P[0], alpha_01_P * lambda_P[1], alpha_10_P* lambda_P[2], alpha_11_P * lambda_P[3]]
    # indices = get_label_distribution_Q(lamb_final, lamb_ori, count_ori)
    #
    # # print(lamb_ori)
    # # print(count_ori)
    # # print(lamb_final)
    # # print(indices)
    #
    # indices_P = group_sample(group_P, [indices[0],indices[1],indices[2],indices[3]])
    # # indices_Q = group_sample(group_Q, [indices[4],indices[5],indices[6],indices[7]])
    #

    # c_1 = alpha_00_P * lambda_P[0] + alpha_01_P * lambda_P[1] + alpha_10_P* lambda_P[2] + alpha_11_P * lambda_P[3]
    # c_2 = alpha_00_Q * lambda_Q[0] + alpha_01_Q * lambda_Q[1] + alpha_10_Q* lambda_Q[2] + alpha_11_Q * lambda_Q[3]
    #
    # # classification_errors.append(1-c_1*(torch.mean(classification_acc_P[indices_P]).item()) - c_2 * (torch.mean(classification_acc_Q[indices_Q]).item()) )
    #
    # # DEBUG
    # classification_errors.append(1 - c_1 * (torch.mean(classification_acc_P[indices_P]).item()) )

# print(distances)
# print(classification_errors)







hellinger_distances = np.linspace(args.certificate_low, args.certificate_high, args.num_points_bound)

N = len(group_00_P) + len(group_01_P) + len(group_10_P) + len(group_11_P)
lambda_00 = 1.0 * len(group_00_P) / N
lambda_01 = 1.0 * len(group_01_P) / N
lambda_10 = 1.0 * len(group_10_P) / N
lambda_11 = 1.0 * len(group_11_P) / N
lambda_P = [lambda_00, lambda_01, lambda_10, lambda_11]
V1, V2 = [], []
N_P = []
N_P.append(len(group_00_P))
N_P.append(len(group_01_P))
N_P.append(len(group_10_P))
N_P.append(len(group_11_P))
V1.append(1-torch.mean(classification_acc_P[group_00_P]).item())
V1.append(1-torch.mean(classification_acc_P[group_01_P]).item())
V1.append(1-torch.mean(classification_acc_P[group_10_P]).item())
V1.append(1-torch.mean(classification_acc_P[group_11_P]).item())

V2.append(torch.var(classification_acc_P[group_00_P]).item())
V2.append(torch.var(classification_acc_P[group_01_P]).item())
V2.append(torch.var(classification_acc_P[group_10_P]).item())
V2.append(torch.var(classification_acc_P[group_11_P]).item())

V1_loss, V2_loss = [], []
V1_loss.append(torch.mean(loss_all_P[group_00_P]).item())
V1_loss.append(torch.mean(loss_all_P[group_01_P]).item())
V1_loss.append(torch.mean(loss_all_P[group_10_P]).item())
V1_loss.append(torch.mean(loss_all_P[group_11_P]).item())
V2_loss.append(torch.var(loss_all_P[group_00_P]).item())
V2_loss.append(torch.var(loss_all_P[group_01_P]).item())
V2_loss.append(torch.var(loss_all_P[group_10_P]).item())
V2_loss.append(torch.var(loss_all_P[group_11_P]).item())



upper_bounds_grammian = None
upper_bounds_label_shifting_finite_sampling = None
upper_bounds_general_shifting_finite_sampling = None


M = args.M
CONFIDENCE_LEVEL = args.confidence



from fairness_bound_general_shifting import fairness_upper_bound_general_shifting
if args.solve_general_shifting_finite_sampling:
    if args.use_loss:
        V1_upper = list(V1_loss)
        V1_lower = list(V1_loss)
        V2_upper = list(V2_loss)
        V2_lower = list(V2_loss)
        V1 = list(V1_loss)
        V2 = list(V2_loss)
    else:
        V1_upper = list(V1)
        V1_lower = list(V1)
        V2_upper = list(V2)
        V2_lower = list(V2)
    lambda_P_upper = list(lambda_P)
    lambda_P_lower = list(lambda_P)

    if args.finite_sampling:
        for i in range(len(V1_upper)):
            V1_upper[i] = min(M, V1_upper[i] + M * np.sqrt(np.log(2 * 12 / CONFIDENCE_LEVEL) / (2 * N_P[i])))
        for i in range(len(V1_lower)):
            V1_lower[i] = min(M, V1_lower[i] + M * np.sqrt(np.log(2 * 12 / CONFIDENCE_LEVEL) / (2 * N_P[i])))
        for i in range(len(V2_upper)):
            V2_upper[i] = (np.sqrt(V2_upper[i]) + M * np.sqrt(2 * np.log(2 * 12 / CONFIDENCE_LEVEL) / (N_P[i] - 1))) ** 2
        for i in range(len(V2_lower)):
            V2_lower[i] = (np.sqrt(V2_lower[i]) - M * np.sqrt(2 * np.log(2 * 12 / CONFIDENCE_LEVEL) / (N_P[i] - 1))) ** 2
        for i in range(len(lambda_P_upper)):
            lambda_P_upper[i] = min(1, lambda_P_upper[i] + np.sqrt(np.log( 2 * 12 / CONFIDENCE_LEVEL) / (2 * N)))
            lambda_P_lower[i] = lambda_P_lower[i] - np.sqrt(np.log( 2 * 12 / CONFIDENCE_LEVEL) / (2 * N))

    # print(f'N_P: {N_P}')
    # print(f'V1: {V1}')
    # print(f'V1_upper: {V1_upper}')
    # print(f'V2: {V2}')
    # print(f'V2_upper: {V2_upper}')
    # print(f'Lambda_P: {lambda_P}')
    # print(f'Lambda_P_upper: {lambda_P_upper}')
    # upper_bounds_general_shifting_finite_sampling = fairness_upper_bound_general_shifting_finite_sampling(hellinger_distances,
    #         lambda_P, V1_upper, V1_lower, V2_upper, V2_lower, args.gamma_k, args.gamma_r, args.interval_kr, lambda_P_lower, lambda_P_upper)
    upper_bounds_general_shifting_finite_sampling = fairness_upper_bound_general_shifting(hellinger_distances, lambda_P, V1, V2, args.gamma_k, args.gamma_r, args.interval_kr)



print(hellinger_distances)
print('upper bounds')
# upper_bounds_general_shifting_finite_sampling = [0.4540662010704407, 0.49075074856727935, 0.5460979699335423, 0.6095292553214888, 0.6773090797122546, 0.7477965068246802, 0.8200616519575968, 0.8936213382900022, 0.9680849870503605, 1.0432352699716076, 1.1188889860997233, 1.194859006678695]
# upper_bounds_general_shifting_finite_sampling = [0.24701569460866202, 0.29591203354240686, 0.3588813018201282, 0.43015243051579394, 0.5067190588966375, 0.5860755874963361, 0.6657468579105873, 0.7432127712998374, 0.8158550700902565, 0.8809835035627527]
print(upper_bounds_general_shifting_finite_sampling)

import plot.plot as plot

err_P = 1 - torch.mean(classification_acc_P).item()
# plot.make_plot(args.use_loss, hellinger_distances, upper_bounds_general_shifting_finite_sampling, None, None, None, distances,
#                classification_errors, err_P, args.path_figure)

if args.use_loss:
    print(f'loss_P: {torch.mean(loss_all_P).item()}')
    plot.make_plot(args.use_loss, hellinger_distances, upper_bounds_general_shifting_finite_sampling, None, None, None, distances,
                   losses_all, torch.mean(loss_all_P).item(), args.path_figure, None, None)
else:
    plot.make_plot(args.use_loss, hellinger_distances, upper_bounds_general_shifting_finite_sampling, None, None, None, distances,
                   classification_errors, err_P, args.path_figure, None, None)








