import torch.nn as nn

activation_functions = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(alpha=1.0)
}


class mlp_gaussian(nn.Module):
    def __init__(self, activation='elu', num_hidden=2):
        super(mlp_gaussian, self).__init__()
        self.num_hidden = num_hidden

        if num_hidden == 1:
            self.linear1 = nn.utils.spectral_norm(nn.Linear(2, 4, bias=False), n_power_iterations=1)
            self.linear2 = nn.utils.spectral_norm(nn.Linear(4, 2, bias=False), n_power_iterations=1)
            self.activation = nn.ReLU() if activation == 'relu' else nn.ELU(alpha=1.0)
        elif num_hidden == 2:
            self.linear1 = nn.utils.spectral_norm(nn.Linear(2, 4, bias=False), n_power_iterations=1)
            self.linear2 = nn.utils.spectral_norm(nn.Linear(4, 2, bias=False), n_power_iterations=1)
            self.linear3 = nn.utils.spectral_norm(nn.Linear(2, 2, bias=False), n_power_iterations=1)
            self.activation = nn.ReLU() if activation == 'relu' else nn.ELU(alpha=1.0)
        else:
            raise ValueError

        if activation == 'relu':
            self.active = nn.ReLU()
        else:
            self.active = nn.ELU()

    def forward(self, inputs):
        x = self.active(self.linear1(inputs))
        x = self.linear2(x)
        if self.num_hidden == 2:
            x = self.active(x)
            x = self.linear3(x)
        else:
            raise ValueError
        return x

    def init_weights_glorot(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)