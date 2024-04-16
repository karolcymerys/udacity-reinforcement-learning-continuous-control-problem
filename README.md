# Udacity Project: Continuous Control

__TODO: Put here animation of trained agent__

## Description

The goal of this project is to resolve __multi-agent__ environment using __Reinforcement Learning Policy-Based Method__.

### Environment

This environment is called _Reacher_ environment. 
It consists of 20 double-joined arms (__agents__). 
The goal of single agent is to move arm to the goal location and keep it there.
Environment is considered as resolved in training process when all the agents get 
__an average score of +30 over 100 consecutive episodes__.


### State

The state space of environment is built from 33-dimension vector that includes 
the agent's position, rotation, velocity and angular velocities of the arm.

### Actions

An agent is responsible for deciding, based on provided environment state space 
what action should be taken to get the best score. 
Agent's action is 4-dimensional vector that describes torque applicable to two joints.
Each element of action's vector should be between `-1` and `1`.

### Reward  

- a reward of +0.1 point is provided for each step when the agent's hand is kept in the goal location

## Algorithm Description

All details related to algorithm utilized to resolve this problem can be found in [Report.md file](./Report.md).

## Structure description

The project contains following files:

| Filename                       | Description                                                                    |
|--------------------------------|--------------------------------------------------------------------------------|
| `ddpg.py`                      | implementation of Deep Deterministic Policy Gradient Algorithm                 |
| `doc`                          | folder that contains docs related files                                        |
| `environment.py`               | wrapper class for _UnityEnvironment_ to simplify interactions with environment |
| `model.py`                     | Pytorch-based implementation of _ policy network_ (agent and critic networks)  |
| `Report.md`                    | doc file that contains utilized algorithm's details                            |  
| `requirements.txt`             | file that contains all required Python dependencies                            |  
| `README.md`                    | doc file that contains project's description                                   | 
| `test.py`                      | Python script that allows to start trained agent                               |
| `train.py`                     | main Python script for training                                                |
| `ddpg_actor_model_weights.pt`  | file that contains weights of agent network (DDPG)                             |
| `ddpg_critic_model_weights.pt` | file that contains weights of critic network (DDPG)                            |

## Try it yourself

### Dependencies

- [Python 3.X](https://www.python.org/downloads/)
- [git](https://git-scm.com/downloads)

First clone repository:

```shell
git clone https://github.com/karolcymerys/udacity-reinforcement-learning-continuous-control-problem.git
```

In order to install all required Python dependencies, please run below command:

```shell
pip install -r requirements.txt
```

Next, install custom _UnityAgents_ and its dependencies:

```shell
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install . 
```

Please note that it might be required to remove packages versions from
`deep-reinforcement-learning/python/requirements.txt` to successfully install these dependencies,
but `protobuf` must be used in version `3.20.0`.

At the end, download _Unity Environment_ for your platform:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Once ZIP package is downloaded, please unzip it and place files in archive to root folder.
Please note, that you may need to adjust `filename` parameter of `ReacherEnvironment` class
(it depends on platform you use).

### Training

Feel free to train your own agents. In order to do that:

1. In `train.py` file adjust hyperparameters
2. Run `python train.py` to start training process
3. Once training is completed, then your network is saved in `model_weights.py` file

### Testing

1. In order to run trained agent, run following command `python test.py`
   (it utilizes model weights from `model_weights.py` file)  