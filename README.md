# Udacity Project: Continuous Control

__TODO: Put here animation of trained agent__

## Description

### Environment

__TODO__

### State

__TODO__

### Action

__TODO__

### Reward

__TODO__

## Algorithm Description

All details related to algorithm utilized to resolve this problem can be found in [Report.md file](./Report.md).

## Structure description

__TODO__

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

Feel free to train your own agent. In order to do that:

1. In `train.py` file adjust hyperparameters
2. Run `python train.py` to start training process
3. Once training is completed, then your network is saved in `model_weights.py` file

### Testing

1. In order to run trained agent, run following command `python test.py`
   (it utilizes model weights from `model_weights.py` file)  