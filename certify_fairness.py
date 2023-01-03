import argparse
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import (accuracy_score)

from datasets import AdultDataset,CompasDataset,CrimeDataset,GermanDataset,HealthDataset,LawschoolDataset
from utils import fairness_metric
from utils import Averagemeter
import distribution_shift.label_shifting as label_shifting
import plot.plot as plot

from fairness_bound_label_shifting_finite_sampling import fairness_certification_bound_label_shifting_finite_sampling
from fairness_bound_general_shifting_finite_sampling import fairness_upper_bound_general_shifting_finite_sampling
from fairness_bound_label_shifting_finite_sampling import fairness_certification_bound_label_shifting_finite_sampling_improve

parser = argparse.ArgumentParser(description='Fairness Certificate')
parser.add_argument('-dataset', type=str, choices=['adult','compas','crime','german','health','lawschool'],
                    default='adult')

parser.add_argument('-protected_att', type=str)
parser.add_argument('-load', type=str)
parser.add_argument('-label', type=str)
parser.add_argument('-transfer', type=bool, default=False)
parser.add_argument('-quantiles', type=bool, default=True)

parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-batch-size', type=int, default=256)
parser.add_argument('-balanced', type=bool, default=False)

parser.add_argument('-num_generated_distributions', type=int, default=0)
parser.add_argument('-num_points_bound', type=int, default=10)
parser.add_argument('-certificate_low', type=float, default=0.1)
parser.add_argument('-certificate_high', type=float, default=0.6)
parser.add_argument('-gamma_k', type=float, default=0.0)
parser.add_argument('-gamma_r', type=float, default=0.0)
parser.add_argument('-interval_kr', type=float, default=0.01)

parser.add_argument('-solve_label_shifting_finite_sampling', type=bool, default=True)
parser.add_argument('-solve_general_shifting_finite_sampling', type=bool, default=False)

parser.add_argument('-loss_function', type=str, default='JSD')
parser.add_argument('-use_loss', type=int, default=0)
parser.add_argument('-M', type=float, default=1.0)

parser.add_argument('-finite_sampling', type=bool, default=True)
parser.add_argument('-confidence', type=float, default=0.1)

parser.add_argument('-path_sampled_distances', type=str, default='./logs/Hellinger_Distance_Q_adult')
parser.add_argument('-path_sampled_loss', type=str, default='./logs/Losses_Q_adult')
parser.add_argument('-path_figure', type=str, default='./data/fig_german_sensitive_shifting_loss')


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

def get_dataset_loader(args):
    if args.dataset == 'adult':
        test_dataset = AdultDataset('test', args)
    elif args.dataset == 'compas':
        test_dataset = CompasDataset('test', args, test_size=0.9)
    elif args.dataset == 'crime':
        test_dataset = CrimeDataset('test', args, test_size=0.9)
    elif args.dataset == 'german':
        test_dataset = GermanDataset('test', args, test_size=0.9)
    elif args.dataset == 'health':
        test_dataset = HealthDataset('test', args, test_size=0.2)
    elif args.dataset == 'lawschool':
        test_dataset = LawschoolDataset('test', args)
    loader_P = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return test_dataset, loader_P

test_dataset, loader_P = get_dataset_loader(args)

def run(model, loader):
    predictions = []
    targets = []
    protects = []
    loss_all = []
    avg_loss = Averagemeter.AverageMeter('avg_loss')
    for data, target, protected in loader:
        data, target, protected = data.to(device), target.to(device), protected.to(device)
        logits = model(data)
        bce_loss = binary_cross_entropy(logits, target.unsqueeze_(-1))
        loss_no_reduct = binary_cross_entropy_no_reduction(logits, target).squeeze_(-1)
        loss_all.append(loss_no_reduct)
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
    loss_all = torch.cat(loss_all)
    classification_acc = (predictions==targets).float()
    print(f'Acc of distribution P: {torch.mean(classification_acc).item()}')

    return avg_loss.avg, loss_all, predictions, protects, classification_acc

loss_Ps = []
distances = []
loss_Qs = []
hellinger_distances = None
upper_bounds = None

model = torch.load(f'logs/{args.dataset}_model')
binary_cross_entropy = nn.BCEWithLogitsLoss()
binary_cross_entropy_no_reduction = nn.BCEWithLogitsLoss(
    reduction='none'
)

loss_P, loss_all, predictions, protects, classification_acc = run(model,loader_P)
print('loss of distribution P: {}'.format(loss_P))
parity = fairness_metric.statistical_parity(predictions, protects)
print(f'Demographic parity of distribution P: {parity}')

dists = []
dists_2 = []
losses = []
losses_2 = []
fairness = []
acc_Qs = []
acc_Qs_2 = []
group_00, group_01, group_10, group_11 = [], [], [], []
for i in range(len(test_dataset.labels)):
    if test_dataset.protected[i] == False and test_dataset.labels[i] == 0:
        group_00.append(i)
    elif test_dataset.protected[i] == False and test_dataset.labels[i] == 1:
        group_01.append(i)
    elif test_dataset.protected[i] == True and test_dataset.labels[i] == 0:
        group_10.append(i)
    elif test_dataset.protected[i] == True and test_dataset.labels[i] == 1:
        group_11.append(i)

print(f'The base rate gap of distribution P: {abs(1.0*len(group_01)/(len(group_00)+len(group_01)) - 1.0*len(group_11)/(len(group_10)+len(group_11)))}')
print(f'The number of instances in distribution P: {len(group_00)+len(group_01)+len(group_10)+len(group_11)}')

seeds = list(range(args.num_generated_distributions))
for i in tqdm(range(args.num_generated_distributions)):
    # generate shifted distribution
    indices_Q, dist = label_shifting.label_shifting_binary_label_binary_attr(seeds[i], test_dataset,group_00,group_01,group_10,group_11,args.gamma_k, args.gamma_r)
    if dist > args.certificate_high - 0.02:
        continue

    loss_Q = torch.mean(loss_all[indices_Q]).item()
    acc_Q = 1-torch.mean(classification_acc[indices_Q]).item()
    acc_Qs.append(acc_Q)

    dists.append(dist)
    losses.append(loss_Q)
    fairness.append(torch.mean(fairness_metric.statistical_parity(predictions[indices_Q], protects[indices_Q])).item())

loss_Ps = loss_P
distances = dists
loss_Qs = losses

acc_Ps = 1-torch.mean(classification_acc).item()

hellinger_distances = np.linspace(args.certificate_low, args.certificate_high, args.num_points_bound)

N = len(group_00) + len(group_01) + len(group_10) + len(group_11)
lambda_00 = 1.0 * len(group_00) / N
lambda_01 = 1.0 * len(group_01) / N
lambda_10 = 1.0 * len(group_10) / N
lambda_11 = 1.0 * len(group_11) / N
lambda_P = [lambda_00, lambda_01, lambda_10, lambda_11]
V1, V2 = [], []
N_P = []
N_P.append(len(group_00))
N_P.append(len(group_01))
N_P.append(len(group_10))
N_P.append(len(group_11))
V1.append(1-torch.mean(classification_acc[group_00]).item())
V1.append(1-torch.mean(classification_acc[group_01]).item())
V1.append(1-torch.mean(classification_acc[group_10]).item())
V1.append(1-torch.mean(classification_acc[group_11]).item())

V2.append(torch.var(classification_acc[group_00]).item())
V2.append(torch.var(classification_acc[group_01]).item())
V2.append(torch.var(classification_acc[group_10]).item())
V2.append(torch.var(classification_acc[group_11]).item())

V1_loss, V2_loss = [], []
V1_loss.append(torch.mean(loss_all[group_00]).item())
V1_loss.append(torch.mean(loss_all[group_01]).item())
V1_loss.append(torch.mean(loss_all[group_10]).item())
V1_loss.append(torch.mean(loss_all[group_11]).item())
V2_loss.append(torch.var(loss_all[group_00]).item())
V2_loss.append(torch.var(loss_all[group_01]).item())
V2_loss.append(torch.var(loss_all[group_10]).item())
V2_loss.append(torch.var(loss_all[group_11]).item())

print(lambda_P)
print(V1)
print(V2)

upper_bounds_label_shifting_finite_sampling = None
upper_bounds_general_shifting_finite_sampling = None

CONFIDENCE_LEVEL = args.confidence
M = args.M

if args.solve_label_shifting_finite_sampling:
    if args.use_loss:
        V1_upper = list(V1_loss)
    else:
        V1_upper = list(V1)
    lambda_P_upper = list(lambda_P)

    if args.finite_sampling and args.dataset not in ['german']:
        for i in range(len(V1_upper)):
            V1_upper[i] = min(M, V1_upper[i] + M * np.sqrt(np.log(16 / CONFIDENCE_LEVEL) / (2 * N_P[i])))
        for i in range(len(lambda_P_upper)):
            lambda_P_upper[i] = min(1, lambda_P_upper[i] + np.sqrt(np.log(16 / CONFIDENCE_LEVEL) / (2 * N)))

        upper_bounds_label_shifting_finite_sampling = fairness_certification_bound_label_shifting_finite_sampling(hellinger_distances,lambda_P_upper, V1_upper,gamma_k=args.gamma_k, gamma_r=args.gamma_r)
    if args.finite_sampling and args.dataset in ['german']:
        for i in range(len(V1_upper)):
            V1_upper[i] = min(M, V1_upper[i] + M * np.sqrt(np.log(8 / CONFIDENCE_LEVEL) / (2 * N_P[i])))
        for i in range(len(lambda_P_upper)):
            lambda_P_upper[i] = min(1, lambda_P_upper[i] + np.sqrt(np.log(8 / CONFIDENCE_LEVEL) / (2 * N)))
        upper_bounds_label_shifting_finite_sampling = fairness_certification_bound_label_shifting_finite_sampling_improve(
            hellinger_distances, lambda_P_upper, V1_upper, gamma_k=args.gamma_k, gamma_r=args.gamma_r)

if args.solve_general_shifting_finite_sampling:
    if args.use_loss:
        V1_upper = list(V1_loss)
        V1_lower = list(V1_loss)
        V2_upper = list(V2_loss)
        V2_lower = list(V2_loss)
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


    print(f'V1: {V1}')
    print(f'V1_upper: {V1_upper}')
    print(f'V2: {V2}')
    print(f'V2_upper: {V2_upper}')
    print(f'N_P: {N_P}')
    upper_bounds_general_shifting_finite_sampling = fairness_upper_bound_general_shifting_finite_sampling(hellinger_distances,
            lambda_P, V1_upper, V1_lower, V2_upper, V2_lower, args.gamma_k, args.gamma_r, args.interval_kr, lambda_P_lower, lambda_P_upper)



print(hellinger_distances)
print('upper bounds')

bound = None
if upper_bounds_label_shifting_finite_sampling is not None:
    print('upper_bounds_label_shifting_finite_sampling: ')
    print(upper_bounds_label_shifting_finite_sampling)
    bound = upper_bounds_label_shifting_finite_sampling
if upper_bounds_general_shifting_finite_sampling is not None:
    print('upper_bounds_general_shifting_finite_sampling: ')
    print(upper_bounds_general_shifting_finite_sampling)
    bound = upper_bounds_general_shifting_finite_sampling

if args.use_loss:
    plot.make_plot(args.use_loss, hellinger_distances, bound, None, None, None, distances,
                   loss_Qs, loss_Ps, args.path_figure, None, None)
else:
    plot.make_plot(args.use_loss, hellinger_distances, bound, None, None, None, distances,
                   acc_Qs, acc_Ps, args.path_figure, None, None)


