import torchaudio

CHUNK_SIZE = 44100 * 10  # 10 seconds of audio

waveform_file_path = 'mixture.wav'

waveform, sample_rate = torchaudio.load(waveform_file_path)
# Print the original waveform size
print(f'Original waveform size: {waveform.size()}')

waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)(waveform)
sample_rate = 44100
# Print the resampled waveform size
print(f'Resampled waveform size: {waveform.size()}')

# Trim the waveform to the first 10 seconds
trimmed_waveform = waveform[:, :CHUNK_SIZE]

# Print the trimmed waveform size
print(f'Trimmed waveform size: {trimmed_waveform.size()}')

torchaudio.save('mixture_trimmed.wav', trimmed_waveform, sample_rate)
