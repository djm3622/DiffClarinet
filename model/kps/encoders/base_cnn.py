from torch import nn


class BaseCNN(nn.Module):
    def __init__(self, encoder_type):
        super().__init__()

        self.encoder_type = encoder_type # "s1, m1, l1"

        if self.encoder_type == "s1":
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 4, kernel_size=33, stride=4, padding=16),
                nn.GELU(),

                nn.Conv1d(4, 8, kernel_size=15, stride=4, padding=7),
                nn.GELU(),

                nn.AdaptiveAvgPool1d(8),

                nn.Flatten(),
                nn.Linear(8 * 8, 16),
                nn.GELU(),
                nn.Linear(16, 1)
            )
        elif self.encoder_type == "m1":
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 4, kernel_size=33, stride=4, padding=16),
                nn.GELU(),

                nn.Conv1d(4, 8, kernel_size=15, stride=4, padding=7),
                nn.GELU(),

                nn.Conv1d(8, 16, kernel_size=9, stride=4, padding=4),
                nn.GELU(),

                nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
                nn.GELU(),

                nn.AdaptiveAvgPool1d(8), # number of output points

                nn.Flatten(),
                nn.Linear(32 * 8, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
        elif self.encoder_type == "l1":
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=65, stride=4, padding=32),
                nn.GELU(),

                nn.Conv1d(8, 16, kernel_size=33, stride=4, padding=16),
                nn.GELU(),

                nn.Conv1d(16, 32, kernel_size=15, stride=4, padding=7),
                nn.GELU(),

                nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
                nn.GELU(),

                nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
                nn.GELU(),

                nn.AdaptiveAvgPool1d(8),

                nn.Flatten(),
                nn.Linear(128 * 8, 128),
                nn.GELU(),
                nn.Linear(128, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )

    def forward(self, x):
        return self.encoder(x)
    
