import numpy as np
import torch
from matplotlib import pyplot as plt

from environment import ReacherEnvironment

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def plot_score(values: np.ndarray[np.float64]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(values)), values)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    env = ReacherEnvironment(device=DEVICE, training_mode=False)
    result = env.reset()
    scores = np.zeros(env.agents_size())

    while True:
        actions = np.random.randn(env.agents_size(), env.action_size())
        actions = np.clip(actions, -1, 1)
        actions = torch.from_numpy(actions).float().to(DEVICE)
        results = env.step(actions)
        scores += result.rewards

        if np.any(results.dones):  # exit loop if episode finished
            break

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    plot_score(scores)
