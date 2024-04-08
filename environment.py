from typing import Union, List

import torch
from unityagents import UnityEnvironment, BrainInfo


class ActionResult:
    def __init__(self,
                 states: Union[torch.FloatTensor, torch.cuda.FloatTensor],
                 rewards: List[float],
                 dones: List[bool]) -> None:
        self.states = states
        self.rewards = rewards
        self.dones = dones

    @staticmethod
    def from_brain_info(brain_info: BrainInfo, device: str):
        return ActionResult(
            torch.from_numpy(brain_info.vector_observations).float().to(device),
            brain_info.rewards,
            brain_info.local_done
        )


class ReacherEnvironment:
    def __init__(self,
                 filename: str = './Reacher.x86_64',
                 seed: int = 0,
                 device='cpu',
                 training_mode: bool = False) -> None:
        self.filename = filename
        self.seed = seed
        self.unity_env = UnityEnvironment(filename, seed=seed)
        self.brain_name = self.unity_env.brain_names[0]
        self.device = device
        self.training_mode = training_mode
        self.agents_no = None

    def action_size(self) -> int:
        return self.unity_env.brains[self.brain_name].vector_action_space_size

    def state_size(self) -> int:
        return self.unity_env.brains[self.brain_name].vector_observation_space_size

    def agents_size(self) -> int:
        if not self.agents_no:
            self.agents_no = len(self.unity_env.reset(train_mode=self.training_mode)[self.brain_name].agents)
        return self.agents_no

    def reset(self) -> ActionResult:
        result = self.unity_env.reset(train_mode=self.training_mode)[self.brain_name]
        if not self.agents_no:
            self.agents_no = len(result.agents)
        return ActionResult.from_brain_info(result, self.device)

    def step(self, action: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> ActionResult:
        actions = torch.clip(action, -1, 1).detach().cpu().numpy()
        return ActionResult.from_brain_info(self.unity_env.step(actions)[self.brain_name], self.device)
