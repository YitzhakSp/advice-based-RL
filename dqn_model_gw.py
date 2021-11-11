import torch
import torch.nn as nn
import torch.nn.functional as F
from other_utils import *

# for gridworld
class CoreNetGw(nn.Module):
    def __init__(self, input_dim='ku'):
        super(CoreNetGw, self).__init__()
        self.L1 = nn.Linear(input_dim,25)
        self.L2 = nn.Linear(25,25)

    def forward(self, x):
        if len(x.shape)==4:
            x=torch.squeeze(x,3)
            x=torch.squeeze(x,1)
            x=x.to(info['DEVICE'])
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        return x

class HeadNetGw(nn.Module):
    def __init__(self, n_actions='ku'):
        super(HeadNetGw, self).__init__()
        self.L1 = nn.Linear(25, n_actions)

    def forward(self, x):
        x = self.L1(x)
        return x

class EnsembleNetGw(nn.Module):
    def __init__(self, n_ensemble, n_actions,input_dim):
        super(EnsembleNetGw, self).__init__()
        self.core_net = CoreNetGw(input_dim=input_dim)
        self.net_list = nn.ModuleList([HeadNetGw(n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads
