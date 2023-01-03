import json

import numpy as np
import os
import random
import matplotlib.pyplot as plt

import matplotlib as mpl
import seaborn as sns

import torch


def init_style(sns_style='whitegrid', font_size_base=16, linewdith_base=1.0, font="Times New Roman"):
    sns.set_style(sns_style)
    colors = sns.color_palette('muted')
    mpl.rcParams["font.family"] = font
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = font_size_base
    mpl.rcParams["grid.linewidth"] = linewdith_base / 2.0
    mpl.rcParams["axes.linewidth"] = linewdith_base
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.
    return colors

# plot params
font_size = 20
linewidth = 1.0
colors = init_style(font_size_base=font_size, linewdith_base=linewidth, sns_style='darkgrid')

SEED = 742

RESULTS_DIR = ''
XLABEL = r'Distribution shift $L_2$-norm  $\||\delta\||_2$'
YLABEL = 'JSD loss'
WRM_CERTIFICATE = 'WRM (Sinha et al., 2018)'
GRAMIAN_METHOD_LEGEND = 'Gramian Certificate'
SAVE_AS = ''

# init seed
np.random.seed(SEED)
random.seed(SEED)

def euclidean_distance_to_hellinger(l2_perturbations, sdev):
    return np.sqrt(1 - np.exp(- l2_perturbations ** 2.0 / (8 * sdev ** 2)))

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



# load data
data = np.load(os.path.join('./logs/', 'gaussian_mixture_data.npy'), allow_pickle=True)[()]
covariates, labels = data['test_data']
logits = data['test_logits']
checkpoint = os.path.join('./logs/model_gaussian_mixture')
sdev = 1.0

# compute loss
criterion = JSDLoss(num_classes=2)
loss = 0.

# distances
euclidean_distances = np.linspace(0, 1.0, 26)
hellinger_distances = euclidean_distance_to_hellinger(euclidean_distances, sdev)
ws_distances = euclidean_distances ** 2


# Run certification of ours
print(covariates)
print(labels)
N_P = [0] * 4
group_00, group_01, group_10, group_11 = [], [], [], []
for i,(x,y) in enumerate(zip(covariates,labels)):
    if x[1] <= 0 and y == 0:
        N_P[0] += 1
        group_00.append(i)
    elif x[1] <= 0 and y == 1:
        N_P[1] += 1
        group_01.append(i)
    elif x[1] > 0 and y == 0:
        N_P[2] += 1
        group_10.append(i)
    elif x[1] > 0 and y == 1:
        N_P[3] += 1
        group_11.append(i)
N = N_P[0] + N_P[1] + N_P[2] + N_P[3]
lambda_P = [1.0*N_P[0]/N, 1.0*N_P[1]/N, 1.0*N_P[2]/N, 1.0*N_P[3]/N]

logits = torch.tensor(logits).cuda()
labels = torch.tensor(labels).cuda()
loss_all = []

from tqdm import tqdm
for i in tqdm(range(len(logits))):
    loss_all.append(criterion(logits[i:i+1], labels[i:i+1]).item())

loss_all = np.array(loss_all)
V1, V2 = [], []
V1.append(np.mean(loss_all[group_00]))
V1.append(np.mean(loss_all[group_01]))
V1.append(np.mean(loss_all[group_10]))
V1.append(np.mean(loss_all[group_11]))
V2.append(np.var(loss_all[group_00],ddof=1))
V2.append(np.var(loss_all[group_01],ddof=1))
V2.append(np.var(loss_all[group_10],ddof=1))
V2.append(np.var(loss_all[group_11],ddof=1))


# print(lambda_P)
# print(V1)
# print(V2)
# upper_bounds_general_shifting = fairness_upper_bound_general_shifting(
#     hellinger_distances,
#     lambda_P, V1, V2, 0, 0, 0.005)
# print(hellinger_distances)
# print(upper_bounds_general_shifting)


V1_upper = list(V1)
V1_lower = list(V1)
V2_upper = list(V2)
V2_lower = list(V2)
lambda_P_upper = list(lambda_P)
lambda_P_lower = list(lambda_P)

M = 1.0
CONFIDENCE_LEVEL=0.01
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

print(f'N_P: {N_P}')
print(f'V1: {V1}')
print(f'V1_upper: {V1_upper}')
print(f'V2: {V2}')
print(f'V2_upper: {V2_upper}')
print(f'Lambda_P: {lambda_P}')
print(f'Lambda_P_upper: {lambda_P_upper}')
from fairness_bound_general_shifting_finite_sampling import fairness_upper_bound_general_shifting_finite_sampling
upper_bounds_general_shifting_finite_sampling = fairness_upper_bound_general_shifting_finite_sampling(hellinger_distances,
        lambda_P, V1_upper, V1_lower, V2_upper, V2_lower, 0.0, 0.0, 0.005, lambda_P_lower, lambda_P_upper)

print(hellinger_distances)
print(upper_bounds_general_shifting_finite_sampling)


# Fairness bound subject to sensitive shifting
# upper_bounds_label_shifting_finite_sampling = fairness_certification_bound_label_shifting_finite_sampling(hellinger_distances,lambda_P_upper, V1_upper,gamma_k=args.gamma_k, gamma_r=args.gamma_r)










