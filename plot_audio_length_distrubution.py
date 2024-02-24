import matplotlib.pyplot as plt
import os
import librosa

root_dir = './dataset/test/'

test_lengths = []
for dir in os.listdir(root_dir):
    if(os.path.isdir(os.path.join(root_dir, dir))):
        song_dir = os.path.join(root_dir, dir)
        for file in os.listdir(song_dir):
            audio, _ = librosa.load(os.path.join(song_dir, file), sr=None)
            test_lengths.append(len(audio))

root_dir = './dataset/train/'

train_lengths = []
for dir in os.listdir(root_dir):
    if(os.path.isdir(os.path.join(root_dir, dir))):
        song_dir = os.path.join(root_dir, dir)
        for file in os.listdir(song_dir):
            audio, _ = librosa.load(os.path.join(song_dir, file), sr=None)
            train_lengths.append(len(audio))

train_mean = sum(train_lengths) / len(train_lengths)
test_mean = sum(test_lengths) / len(test_lengths)

train_max = max(train_lengths)
test_max = max(test_lengths)

train_min = min(train_lengths)
test_min = min(test_lengths)

print("Train max:", train_max)
print("Test max:", test_max)

print("Train min:", train_min)
print("Test min:", test_min)

print("Train mean:", train_mean)
print("Test mean:", test_mean)

# plt.hist(lengths, bins=50)
# plt.show()