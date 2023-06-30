from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict

class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))