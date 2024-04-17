from typing import Union, Tuple, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from environment import ReacherEnvironment
from model import PPONetwork

# https://spinningup.openai.com/en/latest/algorithms/ppo.html
# https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl

BATCH_SIZE = 256


class ProximalPolicyOptimization:
    def __init__(self, env: ReacherEnvironment, device: str = 'cpu') -> None:
        self.env = env
        self.device = device
        self.network = PPONetwork(env.state_size(), env.action_size()).to(device)

    def train(self,
              max_episodes: int = 1_000,
              max_t: int = 500,
              k: int = 5,
              actor_network_lr: float = 1e-4,
              eps: float = 0.2,
              eps_decay: float = 0.999,
              gamma: float = 0.99,
              gae_lambda: float = 0.95) -> None:

        def surrogate_fn(
                _new_policy_probs: [torch.FloatType, torch.cuda.FloatTensor],
                _old_policy_probs: [torch.FloatType, torch.cuda.FloatTensor],
                _advantages: [torch.FloatType, torch.cuda.FloatTensor],
                _eps: float
        ) -> Union[torch.FloatType, torch.cuda.FloatTensor]:
            ratio = torch.exp(_new_policy_probs - _old_policy_probs)
            clipped_ratio = torch.clip(ratio, 1 - _eps, 1 + _eps)

            _advantages_normalized = (_advantages - _advantages.mean()) / (_advantages.std() + 1e-12)
            surrogate_loss = torch.min(ratio * _advantages_normalized, clipped_ratio * _advantages_normalized)
            return -torch.mean(surrogate_loss)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylabel('Score')
        plt.xlabel('Episode #')

        scores = []
        actor_optimizer = Adam(self.network.parameters(), actor_network_lr, eps=1e-5)
        actor_optimizer_scheduler = StepLR(actor_optimizer, step_size=50, gamma=0.5)

        critic_loss = torch.nn.MSELoss()

        with (tqdm(range(1, max_episodes + 1)) as episodes):
            for episode_i in episodes:
                trajectories, total_rewards = self.__collect_trajectories(max_t, gamma, gae_lambda)

                self.network.train()

                for epoch in range(k):
                    sample_indices = torch.permute(torch.arange(trajectories['observations'].shape[0]), dims=[0])

                    for start_idx in range(0, len(sample_indices), BATCH_SIZE):
                        states = trajectories['observations'][sample_indices[start_idx:start_idx + BATCH_SIZE], :]
                        future_rewards = trajectories['future_rewards'][sample_indices[start_idx:start_idx + BATCH_SIZE], :]
                        old_policy_probs = trajectories['log_probs'][sample_indices[start_idx:start_idx + BATCH_SIZE]]
                        advantages = trajectories['advantages'][sample_indices[start_idx:start_idx + BATCH_SIZE]]

                        actions, new_policy_probs, entropy = self.network(states)
                        received_v = self.network.critic(states)

                        actor_optimizer.zero_grad()
                        loss = surrogate_fn(new_policy_probs, old_policy_probs, advantages, eps) + critic_loss(received_v, future_rewards)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                        actor_optimizer.step()

                scores.append(total_rewards.detach().cpu().numpy())
                windowed_score = np.array(scores[-100:])
                episodes.set_postfix({
                    'Current Avg reward': np.mean(windowed_score[-1, :]),
                    'Min reward': np.min(np.mean(windowed_score, axis=0)),
                    'Avg reward': np.mean(np.mean(windowed_score, axis=0)),
                    'Max reward': np.max(np.mean(windowed_score, axis=0)),
                })

                if np.min(np.mean(windowed_score, axis=0)) >= 30.0:
                    break

                eps = max(eps*eps_decay, 0.05)
                actor_optimizer_scheduler.step()

                np_scores = np.array(scores)
                plt.plot(np.arange(len(scores)), np.mean(np_scores, axis=1))
                plt.pause(1e-5)

    def __collect_trajectories(self,
                               max_t: int,
                               gamma: float,
                               gae_lambda: float) -> Tuple[
                                        Union[torch.FloatType, torch.cuda.FloatTensor],
                                        Union[torch.FloatType, torch.cuda.FloatTensor]]:
        no_agents = self.env.agents_size()
        buffer = self.__sample_env(no_agents, max_t)
        buffer = self.__calculate_rewards_to_go(buffer, gamma)
        buffer = self.__calculate_advantages(buffer, gamma, gae_lambda)

        t = buffer[0]['observations'].shape[0]
        trajectories = {
            'observations': torch.cat([buffer[agent_id]['observations'][:t-1] for agent_id in range(no_agents)]),
            'actions': torch.cat([buffer[agent_id]['actions'][:t-1] for agent_id in range(no_agents)]),
            'rewards': torch.cat([buffer[agent_id]['rewards'][:t-1] for agent_id in range(no_agents)]),
            'values': torch.cat([buffer[agent_id]['values'][:t-1] for agent_id in range(no_agents)]),
            'log_probs': torch.cat([buffer[agent_id]['log_probs'][:t-1] for agent_id in range(no_agents)]),
            'advantages': torch.cat([buffer[agent_id]['advantages'][:t-1] for agent_id in range(no_agents)]),
            'future_rewards': torch.cat([buffer[agent_id]['future_rewards'][:t-1] for agent_id in range(no_agents)])
        }
        episode_rewards = torch.stack([buffer[agent_id]['rewards'].sum() for agent_id in range(no_agents)])
        return trajectories, episode_rewards

    def __sample_env(self,
                     no_agents: int,
                     max_t: int) -> Dict[int, Dict[str, Union[torch.FloatType, torch.cuda.FloatTensor]]]:
        buffer = dict({
            agent_i: {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': []
            } for agent_i in range(no_agents)})
        total_reward = np.zeros(no_agents)

        self.network.eval()
        with torch.no_grad():
            previous_response = self.env.reset()
            for t_i in range(max_t):
                actions, log_probs, _ = self.network(previous_response.states)
                response = self.env.step(actions)
                values = self.network.critic(response.states)

                total_reward += response.rewards
                for agent_id, agent_trajectories in buffer.items():
                    agent_trajectories['observations'].append(previous_response.states[agent_id, :])
                    agent_trajectories['actions'].append(actions[agent_id, :])
                    agent_trajectories['rewards'].append(torch.tensor(response.rewards[agent_id]).to(self.device))
                    agent_trajectories['values'].append(values[agent_id, :])
                    agent_trajectories['log_probs'].append(log_probs[agent_id])

                if np.any(response.dones):
                    break

                previous_response = response

        return {
            agent_id: {
                'observations': torch.stack(buffer[agent_id]['observations']),
                'actions': torch.stack(buffer[agent_id]['actions']),
                'rewards': torch.stack(buffer[agent_id]['rewards']).view(len(buffer[agent_id]['rewards']), -1),
                'values': torch.stack(buffer[agent_id]['values']),
                'log_probs': torch.stack(buffer[agent_id]['log_probs'])
            }
            for agent_id in range(no_agents)
        }

    def __calculate_rewards_to_go(self,
                                  buffer: Dict[int, Dict[str, Union[torch.FloatType, torch.cuda.FloatTensor]]],
                                  gamma: float) -> Dict[int, Dict[str, Union[torch.FloatType, torch.cuda.FloatTensor]]]:

        discounts = gamma ** torch.arange(0, buffer[0]['rewards'].shape[0], device=self.device).view(buffer[0]['rewards'].shape[0], 1)
        for agent_id, agent_trajectories in buffer.items():
            rewards = agent_trajectories['rewards'] * discounts
            agent_trajectories['future_rewards'] = rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])

        return buffer

    def __calculate_advantages(self,
                               buffer: Dict[int, Dict[str, Union[torch.FloatType, torch.cuda.FloatTensor]]],
                               gamma: float, gae_lambda: float
                               ) -> Dict[int, Dict[str, Union[torch.FloatType, torch.cuda.FloatTensor]]]:

        discounts = (gamma * gae_lambda) ** torch.arange(0, buffer[0]['values'].shape[0] - 1, device=self.device).view(buffer[0]['values'].shape[0] - 1, 1)
        for agent_id, agent_trajectories in buffer.items():
            delta = agent_trajectories['rewards'][:-1] + gamma * agent_trajectories['values'][1:] - agent_trajectories['values'][:-1]
            agent_trajectories['advantages'] = (delta * discounts).flip(dims=[0]).cumsum(dim=0).flip(dims=[0])

        return buffer

