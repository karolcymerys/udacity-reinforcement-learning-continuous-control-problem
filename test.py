import torch

from ddpg import DDPG
from environment import ReacherEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def ddpg(env: ReacherEnvironment) -> None:
    algorithm = DDPG(env.state_size(), env.action_size(), device=DEVICE)
    algorithm.load_weights('ddpg_actor_model_weights.pth')
    algorithm.test(env)


if __name__ == '__main__':
    env = ReacherEnvironment(device=DEVICE, training_mode=False)
    ddpg(env)
