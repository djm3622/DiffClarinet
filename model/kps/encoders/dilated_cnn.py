from torch import nn


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):
        super().__init__()

        padding = dilation * (kernel_size // 2)

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
        )

        self.skip = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))
    

class ResCNN(nn.Module):
    def __init__(self, encoder_type):
        super().__init__()

        self.encoder_type = encoder_type

        if self.encoder_type == "s1":
            self.encoder = nn.Sequential(
                DilatedBlock(1, 4, kernel_size=33, stride=4, dilation=1),
                DilatedBlock(4, 8, kernel_size=15, stride=4, dilation=1),

                nn.AdaptiveAvgPool1d(8),

                nn.Flatten(),
                nn.Linear(8 * 8, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )

        elif self.encoder_type == "m1":
            self.encoder = nn.Sequential(
                DilatedBlock(1, 4, kernel_size=33, stride=4, dilation=1),
                DilatedBlock(4, 8, kernel_size=15, stride=4, dilation=1),
                DilatedBlock(8, 16, kernel_size=9, stride=4, dilation=1),
                DilatedBlock(16, 32, kernel_size=9, stride=4, dilation=1),

                nn.AdaptiveAvgPool1d(8),

                nn.Flatten(),
                nn.Linear(32 * 8, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        elif self.encoder_type == "l1":
            self.encoder = nn.Sequential(
                DilatedBlock(1, 8, kernel_size=65, stride=4, dilation=1),
                DilatedBlock(8, 16, kernel_size=33, stride=4, dilation=1),
                DilatedBlock(16, 32, kernel_size=15, stride=4, dilation=1),
                DilatedBlock(32, 64, kernel_size=9, stride=4, dilation=1),
                DilatedBlock(64, 128, kernel_size=9, stride=4, dilation=1),

                nn.AdaptiveAvgPool1d(8),

                nn.Flatten(),
                nn.Linear(128 * 8, 128),
                nn.GELU(),
                nn.Linear(128, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

    def forward(self, x):
        return self.encoder(x)