import torch
from torch import nn 

class MLP_7Layers(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(3072, 1024),
            #nn.ReLU(),
            nn.Linear(1024, 512),
            #nn.ReLU(),
            nn.Linear(512, 256),
            #nn.ReLU(),
            nn.Linear(256, 128),
            #nn.ReLU(),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Linear(64, 32),
            #nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, input):
        return self.sequential(input)