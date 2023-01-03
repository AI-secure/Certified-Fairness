import argparse
import copy
import json
import numpy as np
import random
from tqdm import trange
import os

import torch
from torch.utils import data as data_utils
import torch.optim as optim

from datasets import gaussian_mixture
import models

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
set_seed(0)

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

train_dataset = gaussian_mixture.GaussianMixture(1.0, 5000)
test_dataset = gaussian_mixture.GaussianMixture(1.0, 1000000)
train_dataloader = data_utils.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
test_dataloader = data_utils.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

model = models.mlp_gaussian()
model = model.to(device)

criterion = JSDLoss(num_classes=2)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)

def adjust_lr_surrogate(optimizer, lr0, epoch):
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_surrogate_loss(model, x_batch, y_batch, lr0, num_steps, loss_function, gamma):
    z_batch = x_batch.data.clone()
    z_batch = z_batch.cuda() if USE_CUDA else z_batch
    z_batch = torch.autograd.Variable(z_batch, requires_grad=True)

    # run inner optimization
    surrogate_optimizer = optim.Adam([z_batch], lr=lr0)
    surrogate_loss = .0  # phi(theta,z0)
    rho = .0  # E[c(Z,Z0)]
    for t in range(num_steps):
        surrogate_optimizer.zero_grad()
        distance = z_batch - x_batch
        rho = torch.mean((torch.norm(distance.view(len(x_batch), -1), 2, 1) ** 2))
        loss_zt = loss_function(model(z_batch.float()), y_batch)
        surrogate_loss = - (loss_zt - gamma * rho)
        surrogate_loss.backward()
        surrogate_optimizer.step()
        adjust_lr_surrogate(surrogate_optimizer, lr0, t + 1)

    if num_steps==0:
        return 0,0,x_batch

    return surrogate_loss.data, rho.data, z_batch

def train_epoch(model, dataloader, optimizer, criterion, lr_surrogate, steps_surrogate, gamma_surrogate):
    model.train()
    total_losses = []
    surrogate_losses = []
    rho_values = []

    for x_batch, y_batch in dataloader:
        x_batch = torch.autograd.Variable(x_batch).cuda()
        y_batch = torch.autograd.Variable(y_batch).cuda()

        # compute surrogate loss (= inner sup)
        surrogate_loss, rho, z_batch = compute_surrogate_loss(model=model, x_batch=x_batch, y_batch=y_batch,  # noqa
                                                              lr0=lr_surrogate, num_steps=steps_surrogate,
                                                              loss_function=criterion, gamma=gamma_surrogate)

        # run outer optimization step
        optimizer.zero_grad()
        total_loss = criterion(model(z_batch.float()), y_batch)
        total_loss.backward()
        optimizer.step()

        surrogate_losses.append(surrogate_loss)
        total_losses.append(total_loss.data)
        rho_values.append(rho)

    return total_losses, surrogate_losses, rho_values

def evaluate(model, dataloader):
    model.eval()
    counter, acc = .0, .0
    for x_batch, y_batch in dataloader:
        if USE_CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        x_batch = torch.autograd.Variable(x_batch)
        y_batch = torch.autograd.Variable(y_batch)

        out = model(x_batch.float())
        _, predicted = torch.max(out, 1)
        counter += y_batch.size(0)
        acc += float(torch.eq(predicted, y_batch).sum().cpu().data.numpy())

    acc = acc / float(counter) * 100.0
    return acc


def compute_logits(model, dataloader):
    model.eval()
    logits = np.empty(shape=(0, dataloader.dataset.N_CLASSES))
    labels = np.empty(shape=0)
    for i, (x_batch, y_batch) in enumerate(dataloader):
        if USE_CUDA:
            x_batch = x_batch.cuda()

        with torch.no_grad():
            batch_logits = model(x_batch.float()).cpu().numpy()
        logits = np.concatenate([logits, batch_logits])
        labels = np.concatenate([labels, y_batch])

    return logits, labels

# train loop
epoch_bar = trange(10, leave=True)
for epoch in epoch_bar:
    total_losses, surrogate_losses, rho_values = train_epoch(model=model,
                                                             dataloader=train_dataloader,
                                                             optimizer=optimizer,
                                                             criterion=criterion,
                                                             lr_surrogate=0.08,
                                                             steps_surrogate=20,
                                                             gamma_surrogate=2)

    total_loss = torch.mean(torch.FloatTensor(total_losses))  # E(l(theta,Z))
    surrogate_loss = torch.mean(torch.FloatTensor(surrogate_losses))  # E[phi_gamma(theta,Z)]
    distance_loss = torch.mean(torch.FloatTensor(rho_values))  # E[c(Z,Z0)]

    # evaluate train and test accuracy
    acc_train = evaluate(model, train_dataloader)
    acc_test = evaluate(model, test_dataloader)


    # update progress bar
    bar_descr = f"[epoch {epoch}] loss: {total_loss:.3f} train acc: {acc_train:.1f}% test acc: {acc_test:.1f}% "
    bar_descr += f"surrogate loss: {surrogate_loss}, dist loss: {distance_loss}"
    epoch_bar.set_description(bar_descr)
    epoch_bar.refresh()


# compute logits on testing data
test_logits, test_labels = compute_logits(model, test_dataloader)

# compute logits on training data
train_logits, train_labels = compute_logits(model, train_dataloader)

# save data
data = {'test_logits': test_logits,
        'train_logits': train_logits,
        'test_labels': test_labels,
        'train_labels': train_labels,
        'train_data': train_dataset.data,
        'test_data': test_dataset.data}
np.save(os.path.join('./logs/', 'gaussian_mixture_data.npy'), data)

torch.save(model,'./logs/model_gaussian_mixture')

