import os
import torch
import numpy as np
import librosa
import torchaudio
import torchaudio.transforms as T
from torch import nn
from torch.utils.data import Dataset
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder

from torchvision import transforms

TRAIN_MAX_LENGTH = 27719408
TEST_MAX_LENGTH = 18979538
SAMPLE_RATE = 44100
SECONDS = 10    # 10seconds
CHUNK_SIZE = SAMPLE_RATE * SECONDS  


# Get all unique labels in your dataset
all_labels = ['bass', 'drums', 'other', 'vocals']

# Fit a LabelEncoder to the labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
encoded_labels = label_encoder.transform(all_labels)

class SourceSeparationModel(nn.Module):
    def __init__(self, num_sources):
        print("init", num_sources)
        self.num_sources = num_sources
        super(SourceSeparationModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(22050)  # Ensure a fixed output size
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)

        # Define your fully connected layers here...
        self.fc1 = nn.Linear(16 * 2 * 27550, 500)  # input size is 16 * 100 * 2205
        self.fc2 = nn.Linear(500, num_sources * 201 * 2206)

    def forward(self, x):
        # Print the shape of x before the reshaping operation
        print("shape of x", x.shape)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # x = self.pool(nn.functional.relu(self.conv3(x)))

        # Flatten the output from your last convolutional or pooling layer
        x = x.view(x.size(0), -1) 
        
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_sources, 201, 2206)  # Reshape the output to match the target shape
        return x

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print("AudioDataset init")
        self.root_dir = root_dir
        self.transform = transform
        self.chunk_size = CHUNK_SIZE
        self.spectrogram = torchaudio.transforms.Spectrogram()
        self.instruments = ['mixture', 'bass', 'drums', 'vocals', 'other']

        self.dirs = os.listdir(root_dir)
        for dir in self.dirs:
            if(not os.path.isdir(os.path.join(self.root_dir, dir))):
                self.dirs.remove(dir)
        
        self.mixture_list = os.listdir(os.path.join(self.root_dir, self.instruments[0]))
        for mix in self.mixture_list:
            if('mixture' not in mix):
                self.mixture_list.remove(mix)

        if(self.root_dir == './dataset/train'):
            self.max_length = TRAIN_MAX_LENGTH
        else:
            self.max_length = TEST_MAX_LENGTH
        

    def __len__(self):
        return len(self.mixture_list)
    
    def pad_or_truncate(self, x, max_length):
        if len(x) > max_length:
            x = x[:max_length]
        elif len(x) < max_length:
            padding = max_length - len(x)
            x = np.pad(x, (0, padding))
        return x

    def __getitem__(self, idx):
        # Get the song ID and chunk ID from the file name
        song_id, chunk_id = os.path.splitext(self.mixture_list[idx])[0].split('_')

        # Initialize a dictionary to store the audio chunks
        audio_chunks = {}
        instrument_list = []

        # Iterate over each instrument
        for instrument in self.instruments:
            # Construct the file path
            file_path = os.path.join(self.root_dir, instrument, f'{song_id}_{chunk_id}.wav')

            # Load the audio chunk
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
            sample_rate = SAMPLE_RATE

            # Convert waveform to spectrogram
            spectrogram = self.spectrogram(waveform)

             # Pad the spectrogram to the maximum length
            max_length = 2206  # You may need to adjust this value
            if spectrogram.size(2) < max_length:
                spectrogram = nn.functional.pad(spectrogram, (0, max_length - spectrogram.size(2)))


            # Apply any additional transformations
            if self.transform:
                spectrogram = self.transform(spectrogram)

            # Append the audio chunk to the list
            if instrument == 'mixture':
                audio_chunks['mixture'] = spectrogram
            else:
                instrument_list.append(spectrogram)

        # Check the number of channels in 'mixture'
        num_channels = audio_chunks['mixture'].shape[0]
        print(f'Number of channels in mixture: {num_channels}')
    
        # Stack the instrument spectrograms along a new dimension to create a tensor
        print("instrument_list", len(instrument_list))
        audio_chunks['instruments'] = instrument_list
    
        audio_chunks['labels'] = encoded_labels  # Encode the labels

        # Return the audio chunks
        return audio_chunks
    
def collate_fn(data):
    # Flatten the list of data points
    data = [sample for sublist in data for sample in sublist]

    # Separate the mixtures, instruments, and labels
    mixtures = [sample['mixture'] for sample in data]
    instruments = [sample['instruments'] for sample in data]
    labels = [sample['labels'] for sample in data]

    # Stack the mixtures and labels to create batches
    mixtures = torch.stack(mixtures)
    labels = torch.stack([torch.tensor(label) for label in labels])  # Convert each label to a tensor and stack them

    # Stack the instrument chunks along a new dimension to preserve the separation between different instruments
    instruments = torch.stack([torch.stack(instrument_chunks) for instrument_chunks in instruments])

    return {'mixture': mixtures, 'instruments': instruments, 'labels': labels}
