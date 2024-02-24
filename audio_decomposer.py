import torch
import torchaudio
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

x, sr = torchaudio.load("mixed-1.wav") # Load the audio
x = torchaudio.functional.resample(x, sr, 16000) # Resample to 16000
sr = 16000

stft = torchaudio.transforms.Spectrogram(n_fft=1024, power=None)(x)
amp = torch.abs(stft) # this is the Spectrogram we want to decompose.
phase = torch.angle(stft)

R = 30

# If `amp` is a batch of spectrograms
if len(amp.shape) == 3:
    batch_size, rows, cols = amp.shape
    dims = (batch_size, rows, cols)

# If `amp` is a stereo audio
# elif len(amp.shape) == 3:
#     channels, rows, cols = amp.shape
#     # Convert to mono by averaging the channels
#     amp = amp.mean(dim=0)
#     dims = amp.shape

# If `amp` is a mono audio
else:
    rows, cols = amp.shape
    dims = amp.shape

# Randomly initialize the W and H matrices
generator = torch.manual_seed(0)
W = torch.normal(0, 2.5, (rows, R), generator=generator).abs()
H = torch.normal(0, 2.5, (R, cols), generator=generator).abs()

eps = 1e-10
V = amp + eps # We will add epsilon to avoid zeros in our matrix V
MAXITER = 5000
ones = torch.ones(dims)

# We are using GPU acceleration for faster matrix multiplication.
# Skip this line if you do not have GPU.
# W, H, V, ones = W.cuda(), H.cuda(), V.cuda(), ones.cuda()

# Iteratively update H and W
for _ in tqdm(range(MAXITER)):
  WH = W@H + eps
  numerator = W.T@(V/WH)
  denominator = W.T@ones.view_as(WH)
  H *= numerator.squeeze() / (denominator + eps)
  
  WH = W@H + eps
  numerator = (V/WH)@H.T
  denominator = ones.view_as(WH)@H.T
  W *= numerator.squeeze() / (denominator + eps)

# Check the convergence error.
print((W@H - V).mean()) # -2.7256e-09 in our case which is accurate enough.

def separate_source(filters, W, H):
  filtered = W[:,filters]@H[filters,:]
  reconstructed_amp = filtered * torch.exp(1j*phase)
  reconstructed_audio = torchaudio.transforms.InverseSpectrogram(n_fft=1024)(reconstructed_amp)
  return reconstructed_audio

audio = separate_source([0], W, H)

# Drums and Bass Guitar
filters = [0, 2, 4, 6, 7, 16, 18, 19, 27, 29]
drums_and_bass = separate_source(filters, W, H)
print(drums_and_bass.shape)
torchaudio.save('drums_and_base.wav', drums_and_bass, sr)

# Violin
filters = [i for i in range(R) if i not in filters]
violin = separate_source(filters, W, H)
torchaudio.save('violin.wav', violin, sr)
