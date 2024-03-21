import torch
import torch.nn as nn
import torchaudio

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(UpsampleBlock, self).__init__()
        self.convtrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.convtrans(x)))

class WaveUNet(nn.Module):
    def __init__(self, num_sources, num_channels):
        super(WaveUNet, self).__init__()
        # Define the architecture here
        # Downsampling path
        self.down1 = DownsampleBlock(num_channels, 16, 15, padding=7, stride=1)
        self.down2 = DownsampleBlock(16, 32, 15, padding=7, stride=1)
        self.down3 = DownsampleBlock(32, 64, 15, padding=7, stride=1)
        self.down4 = DownsampleBlock(64, 128, 15, padding=7, stride=1)
        
        # Upsampling path
        self.up1 = UpsampleBlock(128, 64, 15, padding=7, stride=1)
        self.up2 = UpsampleBlock(64 + 64, 32, 15, padding=7, stride=1) # +64 for skip connection
        self.up3 = UpsampleBlock(32 + 32, 16, 15, padding=7, stride=1) # +32 for skip connection
        self.up4 = UpsampleBlock(16 + 16, num_sources, 15, padding=7, stride=1)  # +16 for skip connection

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Upsample + skip connections
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        
        return u4

device = torch.device("cpu")
print(f'Using {device} device')

model_dir = "models/torch_v1"
model = torch.load(f"{model_dir}/waveunet_model.pth", map_location=device)

model.to(device)

model.eval()

# Load the test data
test_data_path = "data/256k/flac/test/mixture/0_13.flac"

mixture, _ = torchaudio.load(test_data_path)
mixture = mixture.unsqueeze(0)
print(mixture.shape)

# Predict
mixture = mixture.to(device)
output = model(mixture)
print(output.shape)

# Save the output
output = output.squeeze(0)
print(output.shape)
for i in range(output.shape[0]):
    audio = output[i]
    audio = audio.unsqueeze(0)
    print(audio.shape)
    audio = audio.detach()
    torchaudio.save(f"outputs/output_{i}.flac", audio, sample_rate=44100, format="flac")