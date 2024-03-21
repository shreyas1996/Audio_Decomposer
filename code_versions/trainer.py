
import os
import io
import torch
import numpy as np
# import librosa
import torchaudio
# import torchaudio.transforms as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder

# from torchvision import transforms
import requests
import json
import threading


#


TRAIN_MAX_LENGTH = 27719408
TEST_MAX_LENGTH = 18979538
SAMPLE_RATE = 16000
SECONDS = 1    # 1seconds
CHUNK_SIZE = SAMPLE_RATE * SECONDS
BATCH_SIZE = 256
WORKERS = 10
# with open('pid.txt', 'w') as f:
#     f.write(str(os.getpid()))


# Get all unique labels in your dataset
all_labels = ['bass', 'drums', 'other', 'vocals']

# Fit a LabelEncoder to the labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
encoded_labels = label_encoder.transform(all_labels)




# train_url = 'https://staticfiledatasets.s3.ap-south-1.amazonaws.com/audio_decomposer/preprocessed_data/train_files_list.json'
# test_url = 'https://staticfiledatasets.s3.ap-south-1.amazonaws.com/audio_decomposer/preprocessed_data/test_files_list.json'

train_url = 'train_files_list.json'
test_url = 'test_files_list.json'






class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma*out + x
        return out





class SourceSeparationModel(nn.Module):
    def __init__(self, num_sources, num_channels):
        super(SourceSeparationModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            SelfAttention(16),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SelfAttention(32),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SelfAttention(64),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64*((64//2)//2)//2*((82//2)//2)//2, 128)  # Adjust the input size according to your needs
        self.upsample = nn.Upsample(size=(64, 82))
        self.deconv1 = nn.ConvTranspose2d(128, 32, kernel_size=3, padding=1)
        self.final1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.final2 = nn.Conv2d(16, num_sources, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 128, 1, 1)  # Reshape the tensor to match the shape before the fully connected layer
        x = self.upsample(x)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.final1(x))
        x = self.final2(x)
        return x





class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print("AudioDataset init")
        self.root_dir = root_dir
        self.transform = transform
        self.chunk_size = CHUNK_SIZE
        self.spectrogram = torchaudio.transforms.Spectrogram(power=None)
        # Convert to magnitude spectrogram
        # self.spectrogram_transform = torchaudio.transforms.ComplexNorm(power=2.0)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)
        self.instruments = ['mixture', 'bass', 'drums', 'vocals', 'other']

        # Get the list of urls in the root url
        # self.urls = requests.get(root_dir).json()
        with open(root_dir, 'r') as f:
            self.urls = json.load(f)
        # print("URLS", self.urls)


    def __len__(self):
        return len(self.urls)

    def pad_or_truncate(self, x, max_length):
        if len(x) > max_length:
            x = x[:max_length]
        elif len(x) < max_length:
            padding = max_length - len(x)
            x = np.pad(x, (0, padding))
        return x
    
    def generate_spectrogram(self, waveform):
        # Convert waveform to spectrogram
        # stft = self.spectrogram(waveform)

        # Convert to magnitude spectrogram
        spectrogram = self.mel_spectrogram(waveform)


        # Pad the spectrogram to the maximum length
        max_length = 82
        if spectrogram.size(2) < max_length:
            spectrogram = nn.functional.pad(spectrogram, (0, max_length - spectrogram.size(2)))
        
        return spectrogram

    def __getitem__(self, idx):
        # Download the files from the URL
        url_list = self.urls[idx]
        file_list = []
        file_object = {}
        for url in url_list:
            # Split the URL by '/' and keep the part after 'audio_decomposer'
            parts = url.split('/')
            index = parts.index('audio_decomposer')
            path = '/'.join(parts[index+1:])
            # response = requests.get(url)
            file_name = url.split("/")[-1]
            file_list.append(file_name)
            # file_object[url.split('/')[-2]] = io.BytesIO(response.content)
            file_object[url.split('/')[-2]] = path
            # with open(file_name, 'wb') as f:
            #     f.write(response.content)

        # Initialize a dictionary to store the audio chunks
        audio_chunks = {}
        instrument_list = []

        # Iterate over each instrument
        for instrument in self.instruments:
            # Construct the file path
            # file_path = os.path.join(instrument, f'{song_id}_{chunk_id}.wav')

            # Load the audio chunk
            waveform, sample_rate = torchaudio.load(file_object[instrument], format='wav')

            # Convert waveform to spectrogram
            spectrogram = self.generate_spectrogram(waveform)

            # Apply any additional transformations
            if self.transform:
                spectrogram = self.transform(spectrogram)

            # Append the audio chunk to the list
            if instrument == 'mixture':
                audio_chunks['mixture'] = spectrogram
            else:
                instrument_list.append(spectrogram)

        # Check the number of channels in 'mixture'
        # num_channels = audio_chunks['mixture'].shape[0]
        # print(f'Number of channels in mixture: {num_channels}')

        # Stack the instrument spectrograms along a new dimension to create a tensor
        # print("instrument_list", len(instrument_list))
        audio_chunks['instruments'] = instrument_list

        audio_chunks['labels'] = encoded_labels  # Encode the labels

        # Return the audio chunks
        return audio_chunks



#Instantiate the dataset
train_dataset = AudioDataset(root_dir= train_url)
test_dataset = AudioDataset(root_dir= test_url)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, multiprocessing_context='fork')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, multiprocessing_context='fork')


num_channels = 1  # replace with the number of channels in your task
num_sources = num_channels * 4  # replace with the number of sources in your task
model = SourceSeparationModel(num_sources, num_channels)
num_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if os.path.isfile('model_checkpoint.pth'):
    checkpoint = torch.load('model_checkpoint.pth')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    model.to(device)

    # Adjust as necessary
    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 0  # or your custom start epoch if not resuming


def save_checkpoint(epoch, steps, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'steps': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'model_checkpoint.pth')

# Train the model
num_epochs = 5  # replace with the number of epochs you want to train for
print_every = 2000  # print every 2000 mini-batches
best_val_loss = float('inf')  # initialize best validation loss

for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
    model.train()  # set the model to training mode
    running_loss = 0.0
    steps = []
    for i, batch in enumerate(train_loader, 0):
        inputs = batch['mixture']
        instruments = batch['instruments']
        labels = batch['labels']
        steps.append(i)

        # print('inputs', inputs.shape, type(inputs))
        # print('instruments', len(instruments), type(instruments))
        # print('instrument 0', instruments[0].shape, type(instruments[0]))

        # Concatenate instrument tensors along the channel dimension (assuming it's dimension 1)
        targets = torch.cat(instruments, dim=1)

        inputs = inputs.to(device)
        targets = targets.to(device)  # Ensure targets are also moved

        # forward + backward + optimize
        outputs = model(inputs)
        # print('outputs', outputs.shape, type(outputs))
        # print('targets', targets.shape, type(targets))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Zero the gradient buffers
        optimizer.zero_grad()

        # Print statistics for every iteration
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        #save the state of the model
        # save_checkpoint(epoch, steps, model, optimizer, loss)
        # Create a new thread that will run the save_checkpoint function
        checkpoint_thread = threading.Thread(target=save_checkpoint, args=(epoch, steps, model, optimizer, loss))

        # Start the new thread
        checkpoint_thread.start()

# Wait for the last checkpoint thread to finish before exiting the program
checkpoint_thread.join()


# validation
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    val_loss = 0.0
    for i, batch in enumerate(test_loader, 0):
        inputs = batch['mixture']
        instruments = batch['instruments']
        labels = batch['labels']

        # # Stack instrument tensors along a new dimension to create the targets
        # targets = torch.stack(instruments)

        # Concatenate instrument tensors along the channel dimension (assuming it's dimension 1)
        targets = torch.cat(instruments, dim=1)

        inputs = inputs.to(device)
        targets = targets.to(device)  # Ensure targets are also moved

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

    val_loss /= len(test_loader)
    print(f'Validation Loss: {val_loss:.4f}')

    # save the model if validation loss has decreased
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model, 'new_model_v2.pth')
        best_val_loss = val_loss

print('Finished Training')




# best_val_loss = float('inf')  # initialize best validation loss
# # save the model if validation loss has decreased
# if val_loss < best_val_loss:
#     print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
#     torch.save(model, 'new_model_v2.pth')
#     best_val_loss = val_loss

