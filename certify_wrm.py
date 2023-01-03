from collections.abc import Iterable
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


USE_CUDA = torch.cuda.is_available()


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

def adjust_lr_surrogate(optimizer, lr0, epoch):
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

from tqdm import tqdm
class CertifyWRM:
    SUP_ITERATIONS = 200
    SUP_LR = 0.05
    L_LPLUS1 = 2.0

    def __init__(self, checkpoint, x: np.ndarray, y: np.ndarray, finite_sampling: bool = True, num_hidden: int = 1, confidence = 0.1):
        self.model = torch.load(checkpoint)
        self.loss_fn = JSDLoss(num_classes=2, reduce='mean')
        self.n_samples = len(y)
        self.m0 = 1.0

        # init dataloader
        tensor_x = torch.Tensor(x)
        tensor_y = torch.Tensor(y).to(torch.int64)
        dataset = TensorDataset(tensor_x, tensor_y)
        self.dataloader = DataLoader(dataset, batch_size=128, num_workers=4)

        # compute gamma
        self._gamma = self._compute_gamma()
        print(f'gamma: {self._gamma}')


        # finite sampling error
        if finite_sampling:
            self._fs_err_mean = self.m0 * np.sqrt(np.log(1.0 / confidence) / (2.0 * self.n_samples))
        else:
            self._fs_err_mean = .0

    def certify(self, ws_distances):
        wasserstein_distances = [ws_distances] if not isinstance(ws_distances, Iterable) else ws_distances
        surrogate_loss = self._compute_empirical_surrogate_loss().cpu()
        bounds = []
        # for rho in wasserstein_distances:
        #     bounds.append(self._gamma * rho + surrogate_loss + self._fs_err_mean)
        # bounds = np.array(bounds)
        for rho in wasserstein_distances:
            bounds.append(self._gamma * rho + surrogate_loss + self._fs_err_mean)
        bounds = np.array(bounds)
        return bounds

    def _compute_empirical_surrogate_loss(self):
        # compute surrogate loss
        total_surrogate_loss = .0
        for x_batch, y_batch in tqdm(self.dataloader):
            x_batch = torch.autograd.Variable(x_batch).cuda()
            y_batch = torch.autograd.Variable(y_batch).cuda()
            batch_size = len(y_batch)

            # compute surrogate loss (= inner sup)
            surrogate_loss, _, _ = self._surrogate_loss_batch(x_batch, y_batch)  # loss is sum of individual losses
            total_surrogate_loss -= surrogate_loss * batch_size

        # divide by number of samples
        mean_surrogate_loss = total_surrogate_loss / self.n_samples
        return mean_surrogate_loss

    def _surrogate_loss_batch(self, x_batch, y_batch):
        z_batch = x_batch.data.clone()
        z_batch = z_batch.cuda() if USE_CUDA else z_batch
        z_batch = torch.autograd.Variable(z_batch, requires_grad=True)

        # run inner optimization
        surrogate_optimizer = optim.Adam([z_batch], lr=self.SUP_LR)
        surrogate_loss = .0  # phi(theta,z0)
        rho = .0  # E[c(Z,Z0)]
        for t in range(self.SUP_ITERATIONS):
            surrogate_optimizer.zero_grad()
            distance = z_batch - x_batch
            rho = torch.mean((torch.norm(distance.view(len(x_batch), -1), 2, 1) ** 2))
            loss_zt = self.loss_fn(self.model(z_batch.float()), y_batch)
            surrogate_loss = - (loss_zt - self._gamma * rho)
            surrogate_loss.backward()
            surrogate_optimizer.step()
            adjust_lr_surrogate(surrogate_optimizer, self.SUP_LR, t + 1)

        return surrogate_loss.data, rho.data, z_batch

    def _compute_gamma(self):
        # compute operator norm
        weight_mats = [p.detach() for p in self.model.parameters()]
        operator_norms = []

        print(weight_mats)

        for w in weight_mats:
            # operator_norms.append(torch.linalg.matrix_norm(w, 2).cpu().numpy())
            operator_norms.append(np.linalg.norm(w.cpu().numpy(),2))

        alpha_values = np.cumprod(operator_norms)
        beta = alpha_values[-1] * np.sum(alpha_values)
        gamma = 0.314568 * beta + 0.5 * alpha_values[-1] ** 2

        return gamma

    @property
    def gamma(self):
        return self._gamma
