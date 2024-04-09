from typing import Union

import torch
from torch import nn
from torch.nn.init import xavier_normal_ as init_layer


class DDPGActorNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(DDPGActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=action_size),
            nn.Tanh()
        )

        init_layer(self.mlp[0].weight)
        init_layer(self.mlp[2].weight)
        init_layer(self.mlp[4].weight)

    def forward(self, states: Union[torch.FloatTensor, torch.cuda.FloatTensor]
                ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        return self.mlp(states)


class DDPGCriticNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(DDPGCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.v = nn.Sequential(
            nn.Linear(in_features=state_size + action_size, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

        init_layer(self.v[0].weight)
        init_layer(self.v[2].weight)
        init_layer(self.v[4].weight)

    def forward(self,
                states: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                actions: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        return self.v(torch.cat([states, actions], dim=1))
