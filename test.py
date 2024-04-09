import torch

from ddpg import DDPG
from environment import ReacherEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_EPISODES = 1_000

if __name__ == '__main__':
    env = ReacherEnvironment(device=DEVICE, training_mode=False)
    algorithm = DDPG(env.state_size(), env.action_size(), device=DEVICE)
    algorithm.load_weights('actor_model_weights.pth')
    algorithm.test(env)
