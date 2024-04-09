import torch

from ddpg import DDPG
from environment import ReacherEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_EPISODES = 1_000
T_MAX = 500
ACTOR_MODEL_LR = 1e-4
CRITIC_MODEL_LR = 1e-3
TAU = 1e-3
GAMMA = 0.95
BUFFER_SIZE = 250_000

if __name__ == '__main__':
    env = ReacherEnvironment(device=DEVICE, training_mode=True)
    algorithm = DDPG(env.state_size(), env.action_size(), device=DEVICE)

    algorithm.train(env, N_EPISODES, T_MAX, ACTOR_MODEL_LR, CRITIC_MODEL_LR, TAU, GAMMA, BUFFER_SIZE)
    torch.save(algorithm.actor_network.state_dict(), 'actor_model_weights.pth')
    torch.save(algorithm.critic_network.state_dict(), 'critic_model_weights.pth')
