from typing import Union, Tuple

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn.init import xavier_normal_ as init_layer, normal_


class DDPGActorNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(DDPGActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
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


class PPOActorNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(PPOActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.mu = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_size),
            nn.Tanh()
        )

        normal_(self.mu[0].weight)
        normal_(self.mu[2].weight)
        normal_(self.mu[4].weight)

        self.cov_mat = nn.Parameter(torch.diag(torch.full(size=(action_size,), fill_value=0.5)))

    def forward(self, states: Union[torch.FloatTensor, torch.cuda.FloatTensor]
                ) -> Tuple[
                            Union[torch.FloatType, torch.cuda.FloatTensor],
                            Union[torch.FloatType, torch.cuda.FloatTensor],
                            Union[torch.FloatType, torch.cuda.FloatTensor]
                        ]:
        mean = self.mu(states)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample().clip(-1, 1)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_probs, entropy

    def act(self, states: Union[torch.FloatType, torch.cuda.FloatTensor]) -> Tuple[
                                                                        Union[torch.FloatType, torch.cuda.FloatTensor],
                                                                        Union[torch.FloatType, torch.cuda.FloatTensor],
                                                                        Union[torch.FloatType, torch.cuda.FloatTensor]
                                                                    ]:
        mean = self.mu(states)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample().clip(-1, 1)
        log_probs = dist.log_prob(action)
        return action.detach(), log_probs.detach(), dist.entropy()


class PPOCriticNetwork(nn.Module):

    def __init__(self, state_size: int, seed: int = 0) -> None:
        super(PPOCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.v = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

        normal_(self.v[0].weight)
        normal_(self.v[2].weight)
        normal_(self.v[4].weight)

    def forward(self, states: Union[torch.FloatTensor, torch.cuda.FloatTensor]
            ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        return self.v(states)
