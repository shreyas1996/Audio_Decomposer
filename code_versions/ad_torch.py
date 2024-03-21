import torch
import torchaudio
import json
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

SAMPLE_RATE = 44100
SECONDS = 1    # 1 second
CHUNK_SIZE = SAMPLE_RATE * SECONDS
BATCH_SIZE = 32
WORKERS = 4
EPOCHS = 25


train_url = 'assets/train_files_list.json'
test_url = 'assets/test_files_list.json'

# Custom Dataset
class AudioDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.hop_length = SAMPLE_RATE // 64

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        mixture, _ = torchaudio.load(data_path[0])
        mixture_mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, normalized=True, n_mels=64, hop_length=self.hop_length, n_fft=2048,)(mixture)
        source_mels = []
        for source_path in data_path[1:]:
            source, _ = torchaudio.load(source_path)
            source_mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, normalized=True, n_mels=64, hop_length=self.hop_length, n_fft=2048,)(source)
            source_mels.append(source_mel)
            # source_mels.append(source)
        source_mels = torch.cat(source_mels, dim=0)
        # print("SHAPES", mixture_mel.shape, source_mels.shape)
        return mixture_mel, source_mels
        # return mixture, source_mels


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(UpsampleBlock, self).__init__()
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
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

# Helper function to load and preprocess data
def load_data(data_paths, shuffle=False):
    dataset = AudioDataset(data_paths)
    # dataloader = DataLoader(dataset, batch_sampler=SameSizeBatchSampler(SequentialSampler(dataset), batch_size=BATCH_SIZE, drop_last=False), num_workers=WORKERS, multiprocessing_context='fork')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=WORKERS, multiprocessing_context='fork')
    return dataloader

# Load the data from the JSON file
with open(train_url, 'r') as f:
    train_data_paths = json.load(f)
with open(test_url, 'r') as f:
    test_data_paths = json.load(f)

# Load the datasets
train_dataloader = load_data(train_data_paths, shuffle=True)
test_dataloader = load_data(test_data_paths)

# Model instantiation
num_channels = 1  # replace with the number of channels in your task
num_sources = 4  # replace with the number of sources in your task
model = WaveUNet(num_sources=num_sources, num_channels=num_channels)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print(f'Using {device.type} device')

# Model directory
model_dir = 'models/torch_v2'

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Initialize lists to hold the training and validation losses
train_losses = []
val_losses = []

# Start timing
start_time = time.time()

# Training and validation loop
# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0
#     for batch_idx, (mixture, sources) in enumerate(train_dataloader):
#         mixture, sources = mixture.to(device), sources.to(device)
#         optimizer.zero_grad()
#         output = model(mixture)
#         loss = criterion(output, sources)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * mixture.size(0)
#         if batch_idx % 100 == 0:
#             print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss {loss.item()}')
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#         }, f'{model_dir}/checkpoint/model_checkpoint.pth')
#     epoch_loss = running_loss / len(train_dataloader.dataset)
#     train_losses.append(epoch_loss)

#     # validation
#     model.eval()  # set the model to evaluation mode
#     running_loss = 0.0
#     # with torch.no_grad():
#     # val_loss = 0.0
#     for batch_idx, (mixture, sources) in enumerate(test_dataloader):
#         mixture, sources = mixture.to(device), sources.to(device)
#         outputs = model(mixture)
#         loss = criterion(outputs, sources)
#         running_loss += loss.item() * mixture.size(0)
#         # val_loss += loss.item()
#     epoch_loss = running_loss / len(test_dataloader.dataset)
#     val_losses.append(epoch_loss)

#     # val_loss /= len(test_dataloader)
#     print(f'Validation Loss: {epoch_loss:.4f}')

# Assuming you have training_data and validation_data DataLoader objects
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            dataloader = train_dataloader
            model.train()  # Set model to training mode
        else:
            dataloader = test_dataloader
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, sources in tqdm(dataloader):
            inputs = inputs.to(device)
            sources = sources.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, sources)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            true_classes = torch.argmax(sources.data, dim=1)
            running_corrects += torch.sum(preds == true_classes)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'running_loss': running_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'phase': phase,
            }, f'{model_dir}/checkpoint/model_checkpoint.pth')

        epoch_loss = running_loss / len(dataloader.dataset)
        if(device.type == 'mps'):
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
        else:
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
        if phase == 'train':
            train_losses.append(epoch_loss)
        elif phase == 'val':
            val_losses.append(epoch_loss)
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'running_loss': running_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'phase': phase,
            }, f'{model_dir}/checkpoint/model_checkpoint.pth')

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

# End timing
end_time = time.time()
# Calculate total time

training_time = end_time - start_time
print(f'Finished Training in {training_time} seconds')

# Save the model
torch.save(model, f'{model_dir}/waveunet_model.pth')

def plot_training_and_validation_loss(train_losses, val_losses, model_dir):
    # Plot training & validation loss values
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    plt.savefig(f'{model_dir}/stats/loss_plot.png')

    plt.show()

plot_training_and_validation_loss(train_losses, val_losses, model_dir)
