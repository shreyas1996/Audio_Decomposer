# # Load audio
# waveform, sample_rate = torchaudio.load('audio.wav')

# # Convert to mono
# waveform = torch.mean(waveform, dim=0)

# # Create spectrogram
# spectrogram = torchaudio.transforms.Spectrogram()(waveform)

# # Normalize spectrogram
# spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

# # Add extra dimension for batch and channel
# spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)


# # Forward pass
# output = model(spectrogram)

import audio_decomp_module as adm

# Load the entire model
model = adm.torch.load('new_model.pth')
model.eval()

# Load your input audio file
# sample_rate, audio = wavfile.read('mixed-1.wav')
audio, _ = adm.torchaudio.load('mixture_trimmed.wav')

# Check if the audio is mono
if audio.shape[0] == 1:
    # Duplicate the mono channel to create a stereo sound
    audio = adm.torch.cat((audio, audio))

# Normalize audio to [-1, 1]
# audio = audio / 32768.0

# # Convert audio to tensor
# audio_tensor = torch.from_numpy(audio).float()

# Add batch dimension
audio_tensor = audio.unsqueeze(0)

# Forward pass through the model
output = model(audio_tensor)

# The output is a tensor of shape (batch_size, 8, 44100)
# Since we only have one example in the batch, we can remove the batch dimension
output = output.squeeze(0)

# Convert output tensor to numpy array
output = output.detach().numpy()

# Denormalize output
# output = output * 32768.0

# The output is now a numpy array of shape (8, 44100)
# We can write each channel to a separate WAV file
# for i in range(8):
#     wavfile.write(f'output{i}.wav', sample_rate, output[i])

# Assuming outputs is the output of your model and has shape (8, CHUNK_SIZE)
outputs = output.reshape(-1, 2, adm.CHUNK_SIZE)  # Reshape to (4, 2, CHUNK_SIZE)

# Now outputs[i] is an audio file with 2 channels
for i in range(4):
    output_tensor = adm.torch.from_numpy(outputs[i]).float()
    print(output_tensor)
    adm.torchaudio.save(f'output{i}.wav', output_tensor, adm.SAMPLE_RATE)