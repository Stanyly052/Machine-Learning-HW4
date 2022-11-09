import torch
from torch import nn 


class CNN_7Layers(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        #Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
        # bias=True, padding_mode='zeros', device=None, dtype=None)
        self.sequential_Conv = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 64,
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = 1,
            ),
            #nn.ReLU(),
            #nn.BatchNorm2d(64),
            nn.Conv2d(in_channels = 64,
                      out_channels = 128,
                      kernel_size = (3, 3),
                      stride = 2,
                      padding = 1,
            ),
            #nn.ReLU(),
            #nn.BatchNorm2d(128),
            nn.Conv2d(in_channels = 128,
                      out_channels = 256,
                      kernel_size = (3, 3),
                      stride = 2,
                      padding = 1,
            ),
            #nn.ReLU(),
            #nn.BatchNorm2d(256),
            nn.Conv2d(in_channels = 256,
                      out_channels = 256,
                      kernel_size = (3, 3),
                      stride = 2,
                      padding = 1,
            ),
            #nn.ReLU(),
            #nn.BatchNorm2d(256),
        )

        self.sequential_Linear = nn.Sequential(
            nn.Linear(4096, 1024),
            #nn.Dropout(0.5),
            #nn.ReLU(),
            nn.Linear(1024, 1024),
            #nn.Dropout(0.5),
            #nn.ReLU(),
            nn.Linear(1024, 10),
        )
    
    def forward(self, input):

        output_intermedium = self.sequential_Conv(input).view(64, -1)
        output = self.sequential_Linear(output_intermedium)

        return output