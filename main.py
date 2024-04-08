import numpy as np
import torch
from matplotlib import pyplot as plt

from ddpg import DDPG
from environment import ReacherEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_EPISODES = 1_000
T_MAX = 500


def plot_score(values: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(values)), values)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    env = ReacherEnvironment(device=DEVICE, training_mode=True)
    algorithm = DDPG(env.state_size(), env.action_size(), device=DEVICE)

    scores = algorithm.train(env, N_EPISODES, T_MAX)
    torch.save(algorithm.actor_network.state_dict(), f'actor_model_weights.pth')
    torch.save(algorithm.critic_network.state_dict(), f'critic_model_weights.pth')

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    plot_score(scores)
