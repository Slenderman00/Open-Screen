import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUnetAgnostic(nn.Module):
    def __init__(self, depth=8, init_features=64):
        super(SimpleUnetAgnostic, self).__init__()
        self.depth = depth
        self.init_features = init_features
        self.down_convs = nn.ModuleList()
        self.down_convs_pools = nn.ModuleList()

        # Initial input features should be 3 (for RGB)
        last_layer_features = 3

        # Encoder path
        for i in range(1, depth):
            layer_multiplier = 2 ** (i - 1)
            down_conv = nn.Sequential(
                nn.Conv2d(
                    last_layer_features,
                    init_features * layer_multiplier,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(init_features * layer_multiplier),
                nn.ReLU(inplace=True),
            )
            last_layer_features = init_features * layer_multiplier
            self.down_convs.append(down_conv)
            self.down_convs_pools.append(nn.MaxPool2d(2, 2))

        # Decoder path
        self.up_convs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            layer_multiplier = 2 ** (i - 1)
            if i == depth - 1:
                in_channels = init_features * layer_multiplier * 2
            else:
                in_channels = init_features * layer_multiplier * 3

            out_channels = init_features * layer_multiplier

            up_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.up_convs.append(up_conv)

        # Final layer
        self.final_layer = nn.Conv2d(init_features, 1, kernel_size=3, padding=1)

    def __repr__(self):
        """Create a detailed string representation of the model architecture"""
        s = f"SimpleUnetAgnostic(depth={self.depth}, init_features={self.init_features})\n"
        s += "│\n"

        # Input layer
        s += "├── Input Layer [N, 3, H, W]\n"

        # Encoder path
        s += "│\n├── Encoder Path:\n"
        last_channels = 3
        current_size = "H"
        for i in range(self.depth - 1):
            layer_multiplier = 2**i
            channels = self.init_features * layer_multiplier
            s += f"│   ├── Down Block {i+1}:\n"
            s += f"│   │   ├── Conv2d({last_channels} → {channels}, 3x3) + BN + ReLU\n"
            s += f"│   │   │   └── Output: [N, {channels}, {current_size}, {current_size}]\n"
            s += "│   │   └── MaxPool2d(2x2)\n"
            current_size = f"{current_size}/2"
            s += f"│   │       └── Output: [N, {channels}, {current_size}, {current_size}]\n"
            last_channels = channels

        # Bottleneck
        s += "│\n├── Bottleneck\n"
        s += f"│   └── Features: [N, {last_channels}, {current_size}, {current_size}]\n"

        # Decoder path
        s += "│\n├── Decoder Path:\n"
        for i in range(self.depth - 1):
            layer_multiplier = 2 ** (self.depth - 2 - i)
            channels = self.init_features * layer_multiplier
            current_size = f"({current_size})*2"
            s += f"│   ├── Up Block {i+1}:\n"
            s += "│   │   ├── Upsample(scale_factor=2)\n"
            s += f"│   │   ├── Skip Connection from Down Block {self.depth - 1 - i}\n"
            if i == 0:
                in_channels = last_channels * 2
            else:
                in_channels = last_channels * 3
            s += f"│   │   └── Conv2d({in_channels} → {channels}, 3x3) + BN + ReLU\n"
            s += f"│   │       └── Output: [N, {channels}, {current_size}, {current_size}]\n"
            last_channels = channels

        # Final layer
        s += "│\n├── Final Layer:\n"
        s += f"│   ├── Conv2d({last_channels} → 1, 3x3)\n"
        s += "│   └── Sigmoid\n"
        s += "│\n└── Output: [N, 1, H, W]\n"

        return s

    def forward(self, x):
        # Store intermediate outputs for skip connections
        skip_connections = []

        # Encoder path
        current = x
        for i, (down_conv, pool) in enumerate(
            zip(self.down_convs, self.down_convs_pools)
        ):
            current = down_conv(current)
            skip_connections.append(current)
            current = pool(current)

        # Decoder path
        for i, up_conv in enumerate(self.up_convs):
            current = F.interpolate(
                current, scale_factor=2, mode="bilinear", align_corners=False
            )
            skip_connection = skip_connections[-(i + 1)]
            current = torch.cat([current, skip_connection], dim=1)
            current = up_conv(current)

        # Final layer
        current = self.final_layer(current)
        output = torch.sigmoid(current)

        return output
