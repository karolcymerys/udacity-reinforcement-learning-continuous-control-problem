import torch

from ddpg import DDPG
from environment import ReacherEnvironment
from ppo import ProximalPolicyOptimization

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def ddpg(env: ReacherEnvironment) -> None:
    algorithm = DDPG(env.state_size(), env.action_size(), device=DEVICE)

    algorithm.train(env)
    torch.save(algorithm.actor_network.state_dict(), 'ddpg_actor_model_weights.pth')
    torch.save(algorithm.critic_network.state_dict(), 'ddpg_critic_model_weights.pth')


def ppo(env: ReacherEnvironment) -> None:
    algorithm = ProximalPolicyOptimization(env, device=DEVICE)

    algorithm.train()
    torch.save(algorithm.network.state_dict(), 'ppo_model_weights.pth')


if __name__ == '__main__':
    env = ReacherEnvironment(device=DEVICE, training_mode=True)
    ddpg(env)
