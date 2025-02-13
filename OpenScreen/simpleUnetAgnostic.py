import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUnetAgnostic(nn.Module):
    def __init__(self, depth=8, init_features=64):
        super(SimpleUnetAgnostic, self).__init__()
        self.depth = depth
        self.init_features = init_features

        # Encoder path
        self.down_convs = nn.ModuleList()
        curr_channels = 3

        for i in range(depth):
            out_channels = init_features * (2 ** min(i, 3))  # Limit channel growth
            conv = nn.Sequential(
                nn.Conv2d(curr_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.down_convs.append(conv)
            curr_channels = out_channels

        # Decoder path
        self.up_convs = nn.ModuleList()
        for i in range(depth - 1):
            inp_channels = curr_channels
            out_channels = init_features * (2 ** max(min(depth - 2 - i, 3), 0))
            conv = nn.Sequential(
                nn.Conv2d(inp_channels + out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.up_convs.append(conv)
            curr_channels = out_channels

        # Final layer
        self.final_conv = nn.Conv2d(curr_channels, 1, 1)

    def forward(self, x):
        # Store intermediate outputs for skip connections
        skip_connections = []

        # Encoder path with max pooling
        for i, conv in enumerate(self.down_convs[:-1]):
            x = conv(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)

        # Bottom of U-Net
        x = self.down_convs[-1](x)

        # Decoder path
        for i, conv in enumerate(self.up_convs):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skip_connections[-(i + 1)]

            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return torch.sigmoid(self.final_conv(x))


def __repr__(self):
    s = f"SimpleUnetAgnostic(depth={self.depth}, init_features={self.init_features})\n"
    current_channels = 3
    current_size = "H"

    # Input convolution
    s += f"Input Conv: {current_channels} -> {self.init_features} channels\n"
    current_channels = self.init_features

    # Encoder path
    s += "\nEncoder Path:\n"
    for i in range(self.depth - 1):
        s += f"Down Block {i+1}: {current_channels} -> {current_channels*2} channels, size: {current_size}\n"
        current_channels *= 2
        current_size = f"({current_size}/2)"

    # Decoder path
    s += "\nDecoder Path:\n"
    for i in range(self.depth - 1):
        s += f"Up Block {i+1}: {current_channels} -> {current_channels//2} channels, size: {current_size}*2\n"
        current_channels //= 2
        current_size = f"({current_size})*2"

    s += f"\nFinal Conv: {current_channels} -> 1 channel\n"
    return s
