import math
from typing import Union

import torch
from torch import nn


class DDPGActorNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(DDPGActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_size),
            nn.Tanh()
        )

        self.mlp[0].weight.data.uniform_(-1 / math.sqrt(128), 1 / math.sqrt(128))
        self.mlp[3].weight.data.uniform_(-1 / math.sqrt(128), 1 / math.sqrt(128))
        self.mlp[5].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states: Union[torch.FloatTensor, torch.cuda.FloatTensor]
                ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        return self.mlp(states)


class DDPGCriticNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(DDPGCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.v1 = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.v2 = nn.Sequential(
            nn.Linear(in_features=128 + action_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

        self.v1[0].weight.data.uniform_(-1 / math.sqrt(128), 1 / math.sqrt(128))
        self.v2[0].weight.data.uniform_(-1 / math.sqrt(128), 1 / math.sqrt(128))
        self.v2[2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                states: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        x = self.v1(states)
        return self.v2(torch.cat([x, actions], dim=1))
