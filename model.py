import math
from typing import Union, Tuple

import torch
from torch import nn
from torch.distributions import MultivariateNormal


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


class PPONetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(PPONetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.mu = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_size),
            nn.Tanh()
        )
        self.cov_mat = nn.Parameter(torch.diag(torch.full(size=(action_size,), fill_value=0.5)))

        self.v = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

        self.mu[0].weight.data.uniform_(-1/math.sqrt(state_size), 1/math.sqrt(state_size))
        self.mu[3].weight.data.uniform_(-1/math.sqrt(128), 1/math.sqrt(128))
        self.mu[5].weight.data.uniform_(-1/math.sqrt(128), 1/math.sqrt(128))

        self.v[0].weight.data.uniform_(-1/math.sqrt(state_size), 1/math.sqrt(state_size))
        self.v[3].weight.data.uniform_(-1/math.sqrt(128), 1/math.sqrt(128))
        self.v[5].weight.data.uniform_(-1/math.sqrt(128), 1/math.sqrt(128))

    def forward(self, states: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Tuple[
                            Union[torch.FloatType, torch.cuda.FloatTensor],
                            Union[torch.FloatType, torch.cuda.FloatTensor],
                            Union[torch.FloatType, torch.cuda.FloatTensor]
                        ]:
        dist = self.actor(states)
        action = dist.sample().clip(-1, 1)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_probs, entropy

    def actor(self, states: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> MultivariateNormal:
        mean = self.mu(states)
        return MultivariateNormal(mean, nn.functional.relu(self.cov_mat) + 1e-12)

    def critic(self, states: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatType, torch.cuda.FloatTensor]:
        return self.v(states)

    def act(self, states: Union[torch.FloatType, torch.cuda.FloatTensor]) -> Tuple[
                                                                        Union[torch.FloatType, torch.cuda.FloatTensor],
                                                                        Union[torch.FloatType, torch.cuda.FloatTensor],
                                                                        Union[torch.FloatType, torch.cuda.FloatTensor]
                                                                    ]:
        mean = self.mu(states)
        dist = MultivariateNormal(mean, nn.functional.relu(self.cov_mat) + 1e-10)

        action = dist.sample().clip(-1, 1)
        log_probs = dist.log_prob(action)
        return action.detach(), log_probs.detach(), dist.entropy().detach()