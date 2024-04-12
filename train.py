import torch

from ddpg import DDPG
from environment import ReacherEnvironment
from ppoa import ProximalPolicyOptimizationAlgorithm

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_EPISODES = 1_000
T_MAX = 500
ACTOR_MODEL_LR = 3e-4
CRITIC_MODEL_LR = 1e-3
GAMMA = 0.995


TAU = 1e-3
BUFFER_SIZE = 250_000


def ddpg(env: ReacherEnvironment) -> None:

    algorithm = DDPG(env.state_size(), env.action_size(), device=DEVICE)

    algorithm.train(env, N_EPISODES, T_MAX, ACTOR_MODEL_LR, CRITIC_MODEL_LR, TAU, GAMMA, BUFFER_SIZE)
    torch.save(algorithm.actor_network.state_dict(), 'ddpg_actor_model_weights.pth')
    torch.save(algorithm.critic_network.state_dict(), 'ddpg_critic_model_weights.pth')


K = 5
EPS = 0.2
EPS_DECAY = 0.995


def ppo(env: ReacherEnvironment) -> None:
    algorithm = ProximalPolicyOptimizationAlgorithm(env, device=DEVICE)

    algorithm.train(N_EPISODES, K, ACTOR_MODEL_LR, CRITIC_MODEL_LR, EPS, EPS_DECAY, GAMMA)
    torch.save(algorithm.actor_network.state_dict(), 'ppo_actor_model_weights.pth')
    torch.save(algorithm.critic_network.state_dict(), 'ppo_critic_model_weights.pth')


if __name__ == '__main__':
    env = ReacherEnvironment(device=DEVICE, training_mode=True)
    ppo(env)
