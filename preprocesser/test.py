# import requests
# import torchaudio
# import io

# audio_file_path = 'https://staticfiledatasets.s3.ap-south-1.amazonaws.com/audio_decomposer/preprocessed_data/train/mixture/3_000.wav'

# response = requests.get(audio_file_path)
# audio_file = io.BytesIO(response.content)
# waveform, sample_rate = torchaudio.load(audio_file)

# def generate_spectrogram(waveform):
#         # Convert waveform to spectrogram
#         # stft = self.spectrogram(waveform)

#         # Convert to magnitude spectrogram
#         spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=64, sample_rate=16000)(waveform)

#         return spectrogram


# print(waveform.shape)
# print(sample_rate)


# spectogram = generate_spectrogram(waveform)
# print(spectogram.shape)

import librosa
import subprocess
import os
import json
# mixture_path = 'preprocessed_data/train/mixture/3_000.wav'
# mixture, _ = librosa.load(mixture_path, sr=16000)

# print(mixture.shape)
# mixture_mel = librosa.feature.melspectrogram(mixture, sr=16000, n_fft=1600, hop_length=400)
# print(mixture_mel.shape)


def split_and_resample_audio(input_file, output_dir, song_id, sample_rate=16000, duration=1, compression_rate='16k'):
    try:
        command = ['ffmpeg', '-i', input_file, '-ar', str(sample_rate), '-ac', '1', '-ab', compression_rate , '-f', 'segment', '-segment_time', str(duration), os.path.join(output_dir, str(song_id) + '_%03d.wav')]
        # print('Executing command:', ' '.join(command))
        subprocess.run(command)
    except Exception as e:
        print('Error:', e)

# audio_directory = 'dataset/train/A Classic Education - NightOwl/'
# instruments = ['mixture', 'bass', 'drums', 'vocals', 'other']
# for instrument in instruments:
#     print(f"Loading {instrument}...")
#     audio, sr = librosa.load(f'{audio_directory}{instrument}.wav', sr=None)
#     print(f"Audio shape: {audio.shape}, {audio.size}")
#     print(f"Original sample rate: {sr} Hz")
#     # if sr != 16000:
#     #     audio = librosa.resample(audio, sr, 16000)
#     #     print("Audio resampled to 16000 Hz")
#     #     print(f"New audio shape: {audio.shape}")

#     audio_mel = librosa.feature.melspectrogram(y=audio, sr=sr)
#     audio_stft = librosa.stft(audio)
#     print(f"Mel spectrogram shape: {audio_mel.shape}")
#     print(f"STFT shape: {audio_stft.shape}")
#     # output_dir = f'testing/train/{instrument}'
#     # os.makedirs(output_dir, exist_ok=True)
#     # print('##############################################################################################################')
#     # print(f"Splitting and resampling {instrument}...")
#     # print('##############################################################################################################')
#     # split_and_resample_audio(f'{audio_directory}{instrument}.wav', output_dir, 0, sample_rate=sr, duration=1, compression_rate='16k')
#     # audio_chunk_paths = [os.path.join(output_dir, file) for file in os.listdir(output_dir)]
#     # for chunk_path in audio_chunk_paths:
#     #     chunk_audio, chunk_sr = librosa.load(chunk_path, sr=None)
#     #     chunk_duration = len(chunk_audio) / chunk_sr
#     #     print(f"Chunk duration: {chunk_duration} seconds, sample rate: {chunk_sr} Hz")

# audio_path = 'data/256k/flac/train/mixture/0_0.flac'
# audio, sr = librosa.load(audio_path, sr=None)
# print(f"Audio shape: {audio.shape}, {audio.size}")

# audio_mel = librosa.feature.melspectrogram(y=audio, sr=sr)
# print(f"Mel spectrogram shape: {audio_mel.shape}")
        
def get_unique_shapes(dataset):
    unique_shapes_1 = set()
    unique_shapes_2 = set()

    for element in dataset:
        # Add the shape to the set of unique shapes
        unique_shapes_1.add(element[0].shape[0])
        unique_shapes_2.add(element[0].shape[1])

    return (unique_shapes_1, unique_shapes_2)
SAMPLE_RATE = 44100
def load_and_preprocess_data(data_paths):
    try:
        # print("Data paths:", data_paths)
        # data_paths = data_paths.numpy().tolist()  # convert numpy array to list
        # Decode bytes to strings
        # paths = [path.decode('utf-8') for path in data_paths]

        # paths = get_true_file_paths(data_paths)
        mixture_path, bass_path, drums_path, vocals_path, others_path = data_paths

        mixture, _ = librosa.load(mixture_path, sr=SAMPLE_RATE)
        bass, _ = librosa.load(bass_path, sr=SAMPLE_RATE)
        drums, _ = librosa.load(drums_path, sr=SAMPLE_RATE)
        vocals, _ = librosa.load(vocals_path, sr=SAMPLE_RATE)
        other, _ = librosa.load(others_path, sr=SAMPLE_RATE)

        # Compute the Mel spectrogram here
        mixture_mel = librosa.feature.melspectrogram(y=mixture, sr=SAMPLE_RATE)
        bass_mel = librosa.feature.melspectrogram(y=bass, sr=SAMPLE_RATE)
        drums_mel = librosa.feature.melspectrogram(y=drums, sr=SAMPLE_RATE)
        vocals_mel = librosa.feature.melspectrogram(y=vocals, sr=SAMPLE_RATE)
        other_mel = librosa.feature.melspectrogram(y=other, sr=SAMPLE_RATE)

        sources_mel = [bass_mel, drums_mel, vocals_mel, other_mel]

        return mixture_mel, sources_mel
    except Exception as e:
        print('What is this error anyways')
        print('#'* 100)
        print(e)

with open('assets/train_files_list.json', 'r') as f:
    train_data_paths = json.load(f)
with open('assets/test_files_list.json', 'r') as f:
    test_data_paths = json.load(f)

dataset = []
for train_data in train_data_paths:
    dataset.append(load_and_preprocess_data(train_data))

# print('Train Dataset length:', len(dataset))
train_unique_shapes = get_unique_shapes(dataset)

dataset = []
for test_data in test_data_paths:
    dataset.append(load_and_preprocess_data(test_data))

# print('Test Dataset length:', len(dataset))
test_unique_shapes = get_unique_shapes(dataset)

print(train_unique_shapes)
print(test_unique_shapes)
