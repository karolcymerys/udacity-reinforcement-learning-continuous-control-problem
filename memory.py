import random
from collections import deque
from typing import Union, Tuple, List

import torch


class ReplayBuffer:

    def __init__(self,
                 buffer_size: int,
                 device: str,
                 agents: int = 20,
                 seed: int = 0) -> None:
        self.device = device
        random.seed(seed)

        self.state_memory = deque(maxlen=buffer_size)
        self.actions_memory = deque(maxlen=buffer_size)
        self.rewards_memory = deque(maxlen=buffer_size)
        self.next_state_memory = deque(maxlen=buffer_size)
        self.dones_memory = deque(maxlen=buffer_size)
        self.choices = set()

    def add(self,
            states: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            actions: List[float],
            rewards: List[float],
            next_states: Union[torch.FloatTensor, torch.cuda.FloatTensor],
            dones: List[bool]) -> None:
        for agent_id in range(len(actions)):
            self.state_memory.append(states[agent_id, :])
            self.actions_memory.append(actions[agent_id])
            self.rewards_memory.append(rewards[agent_id])
            self.next_state_memory.append(next_states[agent_id, :])
            self.dones_memory.append(dones[agent_id])
            self.choices.add(self.__len__() - 1)

    def sample(self, batch_size: int) -> Tuple[
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.FloatTensor, torch.cuda.FloatTensor],
        Union[torch.IntTensor, torch.cuda.IntTensor]
    ]:
        selected_indices = random.sample(self.choices,
                                         self.__len__() if self.__len__() < batch_size else batch_size)

        return (
            torch.stack([self.state_memory[index] for index in selected_indices], dim=0).float().to(self.device),
            torch.stack([self.actions_memory[index] for index in selected_indices]).float().to(self.device),
            torch.tensor([[self.rewards_memory[index]] for index in selected_indices]).float().to(self.device),
            torch.stack([self.next_state_memory[index] for index in selected_indices], dim=0).float().to(self.device),
            torch.tensor([[self.dones_memory[index]] for index in selected_indices]).int().to(self.device)
        )

    def __len__(self) -> int:
        return len(self.state_memory)
