import random
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.init import xavier_normal_ as init_layer
from tqdm import tqdm

from environment import ReacherEnvironment
from memory import ReplayBuffer


class ActorNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(ActorNetwork, self).__init__()
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


class CriticNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed: int = 0) -> None:
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.v = nn.Sequential(
            nn.Linear(in_features=state_size+action_size, out_features=128),
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


class DDPG:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int = 0,
                 device: str = 'cpu',
                 training_mode: bool = False) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.training_mode = training_mode

        self.actor_network = ActorNetwork(state_size, action_size).to(device)
        self.critic_network = CriticNetwork(state_size, action_size).to(device)

        random.seed(seed)

    def act(self,
            states: Union[torch.FloatType, torch.cuda.FloatTensor],
            noise_scale: float = 0.0) -> Union[torch.FloatType, torch.cuda.FloatTensor]:
        self.actor_network.eval()
        with torch.no_grad():
            actions = self.actor_network(states) + \
                      noise_scale * torch.randn(states.shape[0], self.action_size).to(self.device)  # Gaussian noise

        return torch.clip(actions, -1, 1)

    def train(self,
              env: ReacherEnvironment,
              max_episodes: int,
              max_t: int = 500,
              actor_lr: float = 3e-4,
              critic_lr: float = 3e-3,
              tau: float = 1e-3,
              gamma: float = 0.95,
              buffer_size: int = 100_000,
              noise_factor: float = 0.1,
              noise_factor_decay: float = 0.995) -> np.ndarray:
        # https://spinningup.openai.com/en/latest/algorithms/ddpg.html

        target_actor_network = ActorNetwork(self.state_size, self.action_size).to(self.device)
        target_critic_network = CriticNetwork(self.state_size, self.action_size).to(self.device)
        self.__soft_update(self.actor_network, target_actor_network, 1.0)
        self.__soft_update(self.critic_network, target_critic_network, 1.0)

        actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), actor_lr)
        critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), critic_lr)

        replay_buffer = ReplayBuffer(buffer_size, self.device)
        scores = []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylabel('Score')
        plt.xlabel('Episode #')

        with (tqdm(range(1, max_episodes + 1)) as episodes):
            for episode_i in episodes:

                previous_results = env.reset()
                total_reward = np.zeros(env.agents_size())
                for t in range(1, max_t + 1):
                    actions = self.act(previous_results.states, noise_factor)
                    results = env.step(actions)

                    replay_buffer.add(
                        previous_results.states,
                        actions,
                        results.rewards,
                        results.states,
                        results.dones
                    )
                    total_reward += results.rewards

                    if t % 20 == 0:
                        self.__learn(
                            target_actor_network,
                            target_critic_network,
                            actor_optimizer,
                            critic_optimizer,
                            replay_buffer,
                            gamma,
                            tau
                        )

                    if np.any(results.dones):
                        break

                    previous_results = results

                scores.append(total_reward)
                windowed_score = np.array(scores[-100:])
                episodes.set_postfix({
                    'Current Avg reward': np.mean(windowed_score[-1, :]),
                    'Min reward': np.min(np.mean(windowed_score, axis=0)),
                    'Avg reward': np.mean(np.mean(windowed_score, axis=0)),
                    'Max reward': np.max(np.mean(windowed_score, axis=0)),
                })

                if np.min(np.mean(windowed_score, axis=0)) >= 30.0:
                    break

                noise_factor = max(noise_factor*noise_factor_decay, 0.001)

                np_scores = np.array(scores)
                for i in range(20):
                    plt.plot(np.arange(len(scores)), np_scores[:, i], label=f'Agent {i}')
                plt.pause(1e-5)

        return np.mean(scores, axis=0)

    def __learn(self,
                target_actor_network: torch.nn.Module,
                target_critic_network: torch.nn.Module,
                actor_optimizer: torch.optim.Optimizer,
                critic_optimizer: torch.optim.Optimizer,
                replay_buffer: ReplayBuffer,
                gamma: float,
                tau: float) -> None:
        # https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

        self.actor_network.train()
        target_actor_network.train()
        self.critic_network.train()
        target_critic_network.train()
        critic_loss_fn = nn.MSELoss()

        for agent_id in range(10):
            samples = replay_buffer.sample(256)

            expected_actions = target_actor_network(samples[3])
            expected_q_value = target_critic_network(samples[3], expected_actions)
            expected_y = samples[2] + (gamma * (1 - samples[4]) * expected_q_value)

            y = self.critic_network(samples[0], samples[1])
            critic_loss = critic_loss_fn(y, expected_y.detach())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1)
            critic_optimizer.step()

            action = self.actor_network(samples[0])
            actor_loss = -(self.critic_network(samples[0], action)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
            actor_optimizer.step()

            self.__soft_update(self.actor_network, target_actor_network, tau)
            self.__soft_update(self.critic_network, target_critic_network, tau)

    @staticmethod
    def __soft_update(src_network: torch.nn.Module, desc_network: torch.nn.Module, tau: float) -> None:
        for desc_param, src_param in zip(desc_network.parameters(), src_network.parameters()):
            desc_param.data.copy_(tau * src_param.data + (1.0 - tau) * desc_param.data)
