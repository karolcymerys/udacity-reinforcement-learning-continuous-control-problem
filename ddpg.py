import copy
import random
from typing import Union, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from environment import ReacherEnvironment
from memory import ReplayBuffer
from model import DDPGActorNetwork, DDPGCriticNetwork


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    # https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3

    def __init__(self, size: Tuple[int, int], mu: float = 0., theta: float = 0.15, sigma: float = 0.2, device: str = 'cpu') -> None:
        """Initialize parameters and noise process."""
        self.mu = mu * torch.ones(*size, device=device)
        self.theta = theta
        self.sigma = sigma
        self.state = mu * torch.ones(*size, device=device)
        self.device = device

    def step(self) -> None:
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> Union[torch.FloatType, torch.cuda.FloatTensor]:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(*self.mu.shape, device=self.device)
        self.state = x + dx
        return self.state


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

        self.actor_network = DDPGActorNetwork(state_size, action_size).to(device)
        self.critic_network = DDPGCriticNetwork(state_size, action_size).to(device)

        random.seed(seed)

    def act(self,
            states: Union[torch.DoubleTensor, torch.cuda.DoubleTensor],
            noise_sampler: OUNoise) -> Union[torch.DoubleTensor, torch.cuda.DoubleTensor]:
        self.actor_network.eval()
        with torch.no_grad():
            ou_noise = noise_sampler.sample()
            return (self.actor_network(states) + ou_noise).clip(-1, 1)

    def train(self,
              env: ReacherEnvironment,
              max_episodes: int = 1_000,
              max_t: int = 500,
              actor_lr: float = 1e-4,
              critic_lr: float = 3e-4,
              tau: float = 1e-3,
              gamma: float = 0.95,
              buffer_size: int = 100_000,
              optimizer_timestamps: int = 10,
              minibatch_size: int = 128) -> None:
        # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        # https://arxiv.org/abs/1509.02971

        target_actor_network = DDPGActorNetwork(self.state_size, self.action_size).to(self.device)
        target_critic_network = DDPGCriticNetwork(self.state_size, self.action_size).to(self.device)
        self.__soft_update(self.actor_network, target_actor_network, 1.0)
        self.__soft_update(self.critic_network, target_critic_network, 1.0)
        noise_sampler = OUNoise((env.agents_size(), self.action_size), device=self.device)

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
                    actions = self.act(previous_results.states, noise_sampler)
                    results = env.step(actions)

                    replay_buffer.add(
                        previous_results.states,
                        actions,
                        results.rewards,
                        results.states,
                        results.dones
                    )
                    total_reward += results.rewards

                    if t % optimizer_timestamps == 0:
                        self.__learn(
                            target_actor_network,
                            target_critic_network,
                            actor_optimizer,
                            critic_optimizer,
                            replay_buffer,
                            gamma,
                            tau,
                            minibatch_size
                        )

                    if np.any(results.dones):
                        break

                    previous_results = results

                scores.append(np.mean(total_reward))
                episodes.set_postfix({
                    'Current Avg reward': np.mean(scores[-1]),
                    'Avg reward': np.mean(scores[-100:]),
                })

                if np.mean(scores[-100:]) >= 30.0:
                    break

                noise_sampler.step()
                np_scores = np.array(scores)
                plt.plot(np.arange(len(scores)), np_scores)
                plt.pause(1e-5)

    def __learn(self,
                target_actor_network: torch.nn.Module,
                target_critic_network: torch.nn.Module,
                actor_optimizer: torch.optim.Optimizer,
                critic_optimizer: torch.optim.Optimizer,
                replay_buffer: ReplayBuffer,
                gamma: float,
                tau: float,
                minibatch_size: int) -> None:
        # https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

        self.actor_network.train()
        target_actor_network.train()
        self.critic_network.train()
        target_critic_network.train()
        critic_loss_fn = nn.MSELoss()

        for _ in range(20):
            samples = replay_buffer.sample(minibatch_size)
            states, actions, rewards, next_states, dones = samples

            expected_actions = target_actor_network(next_states)
            expected_q_value = target_critic_network(next_states, expected_actions)
            expected_y = rewards + (gamma * (1 - dones) * expected_q_value)

            y = self.critic_network(states, actions)
            critic_loss = critic_loss_fn(y, expected_y.detach())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1)
            critic_optimizer.step()

            action = self.actor_network(states)
            actor_loss = -(self.critic_network(states, action)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
            actor_optimizer.step()

            self.__soft_update(self.actor_network, target_actor_network, tau)
            self.__soft_update(self.critic_network, target_critic_network, tau)

    @staticmethod
    def __soft_update(src_network: torch.nn.Module, desc_network: torch.nn.Module, tau: float) -> None:
        for src_param, desc_param in zip(src_network.parameters(), desc_network.parameters()):
            desc_param.data.copy_(tau * src_param.data + (1.0 - tau) * desc_param.data)

    def load_weights(self, actor_weights_filepath: str) -> None:
        self.actor_network.load_state_dict(torch.load(actor_weights_filepath))

    def test(self, env: ReacherEnvironment) -> None:
        results = env.reset()
        scores = np.zeros(env.agents_size())
        while not np.any(results.dones):
            actions = self.act(results.states)
            results = env.step(actions)
            scores += results.rewards
            print(f'\rCurrent score: {np.mean(scores)}', end='')
