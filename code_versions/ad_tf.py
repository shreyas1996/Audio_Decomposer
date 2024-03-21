import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
import json
import librosa
import matplotlib.pyplot as plt

print("CUDA LIST DEVICES") 
print(tf.config.list_physical_devices('GPU'))

# Set the policy
mixed_precision.set_global_policy('mixed_float16')

# Constants
TRAIN_MAX_LENGTH = 27719408
TEST_MAX_LENGTH = 18979538
SAMPLE_RATE = 44100
SECONDS = 1    # 1seconds
CHUNK_SIZE = SAMPLE_RATE * SECONDS
BATCH_SIZE = 16
WORKERS = 10
EPOCHS = 10
BUFFER_SIZE = 10000

all_labels = ['bass', 'drums', 'other', 'vocals']

train_url = 'assets/train_files_list.json'
test_url = 'assets/test_files_list.json'

# Self-Attention Layer
class SelfAttention(layers.Layer):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = layers.Conv2D(in_dim//8, 1)
        self.key_conv = layers.Conv2D(in_dim//8, 1)
        self.value_conv = layers.Conv2D(in_dim, 1)
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros')

    def call(self, x):
        batch_size, width, height, C = x.shape
        query = self.query_conv(x)
        query = tf.reshape(query, [batch_size, -1, width*height])
        query = tf.transpose(query, perm=[0, 2, 1])
        key = self.key_conv(x)
        key = tf.reshape(key, [batch_size, -1, width*height])
        energy = tf.matmul(query, key)
        attention = tf.nn.softmax(energy, axis=-1)
        value = self.value_conv(x)
        value = tf.reshape(value, [batch_size, -1, width*height])
        out = tf.matmul(value, tf.transpose(attention, perm=[0, 2, 1]))
        out = tf.reshape(out, [batch_size, width, height, C])
        out = self.gamma * out + x
        return out

# Model
class SourceSeparationModel(Model):
    def __init__(self, num_sources, num_channels, input_shape):
        super(SourceSeparationModel, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=input_shape)
        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(16, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            # SelfAttention(16),
            layers.MaxPooling2D(2)
        ])
        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(32, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            # SelfAttention(32),
            layers.MaxPooling2D(2)
        ])
        self.conv3 = tf.keras.Sequential([
            layers.Conv2D(128, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            # SelfAttention(128),
            layers.MaxPooling2D(2)
        ])
        self.global_avg_pool = layers.GlobalAveragePooling2D()  # Replace Flatten with GlobalAveragePooling2D
        self.fc = layers.Dense(128)
        self.upsample = layers.UpSampling2D(size=(128, 82))
        self.deconv1 = layers.Conv2DTranspose(32, 3, padding='same')
        self.final1 = layers.Conv2D(16, 3, padding='same')
        self.final2 = layers.Conv2D(num_sources, 3, padding='same')

    def call(self, x):
        x = self.input_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)  # Use GlobalAveragePooling2D here
        x = self.fc(x)
        x = tf.reshape(x, [-1, 1, 1, 128])
        x = self.upsample(x)
        x = tf.nn.relu(self.deconv1(x))
        x = tf.nn.relu(self.final1(x))
        x = self.final2(x)
        return x

# Model V2
class WaveUNet(tf.keras.Model):
    def __init__(self, num_sources):
        super(WaveUNet, self).__init__()

        # Define the downsampling path (contracting path)
        self.downsampling_layers = [
            self.conv_block(8, 5),
            self.conv_block(16, 5),
            self.conv_block(32, 5),
            self.conv_block(64, 5),
            # self.conv_block(128, 5),
        ]

        # Define the bottleneck layer
        self.bottleneck = self.conv_block(128, 5)

        # Define the upsampling path (expansive path)
        self.upsampling_layers = [
            # self.upconv_block(256, 5),
            self.upconv_block(64, 5),
            self.upconv_block(32, 5),
            self.upconv_block(16, 5),
            self.upconv_block(8, 5),
        ]

        # Define the final output layer
        self.final = tf.keras.layers.Conv2D(num_sources, 1)

    def conv_block(self, filters, kernel_size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.MaxPooling2D(2)
        ])

    def upconv_block(self, filters, kernel_size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.UpSampling2D(2)
        ])

    def call(self, x):
        # print("X SHAPE", x.shape)
        skips = []
        for down in self.downsampling_layers:
            x = down(x)
            #  print("Downsampling x shape:", x.shape)
            skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.upsampling_layers, reversed(skips)):
            x = up(x)
            #  print("Upsampling x shape:", x.shape)
            #  print("Skip shape:", skip.shape)
            x = tf.concat([x, skip], axis=-1)

        return self.final(x)

# Data loading and preprocessing is not directly translatable from PyTorch to TensorFlow.
# You would need to write custom data loading and preprocessing code using TensorFlow's tf.data API.
# Here is a placeholder for the data loading part.
    
def get_true_file_paths(urls):
    paths = []
    for url in urls:
        parts = url.split('/')
        index = parts.index('audio_decomposer')
        path = '/'.join(parts[index+1:])
        paths.append(path)
    return paths

def element_length_func(tensor1, tensor2):
    # tf.print("TENSOR 1 SHAPE", tf.shape(tensor1))
    # tf.print("TENSOR 2 SHAPE", tf.shape(tensor2))

    return tf.shape(tensor1)[1]

def pad_mel_spectrogram(mel_spectrogram):
        # Determine the size of the padding
        padding_size = 82 - tf.shape(mel_spectrogram)[1]

        # If the padding size is less than 0, set it to 0
        padding_size = tf.maximum(padding_size, 0)

        # Pad the Mel spectrogram
        mel_spectrogram = tf.pad(mel_spectrogram, paddings=[[0, 0], [0, padding_size]], mode='CONSTANT', constant_values=0)

        return mel_spectrogram

def load_and_preprocess_data(data_paths):
    try:
        # print("Data paths:", data_paths)
        data_paths = data_paths.numpy().tolist()  # convert numpy array to list
        # Decode bytes to strings
        paths = [path.decode('utf-8') for path in data_paths]

        # paths = get_true_file_paths(data_paths)
        mixture_path, bass_path, drums_path, vocals_path, others_path = paths

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

        # Add an extra channel dimension to the Mel spectrograms
        mixture_mel = tf.expand_dims(mixture_mel, axis=-1)
        # mixture_mel.set_shape((None, None, 1))

        bass_mel = tf.expand_dims(bass_mel, axis=-1)
        drums_mel = tf.expand_dims(drums_mel, axis=-1)
        vocals_mel = tf.expand_dims(vocals_mel, axis=-1)
        other_mel = tf.expand_dims(other_mel, axis=-1)

        sources_mel = tf.concat([bass_mel, drums_mel, vocals_mel, other_mel], axis=-1)
        # sources_mel.set_shape((None, None, 4))

        return mixture_mel, sources_mel
    except Exception as e:
        print('What is this error anyways')
        print('#'* 100)
        print(e)

def tf_load_and_preprocess_data(data_paths):
    [mixture_mel, sources_mel] = tf.py_function(load_and_preprocess_data, [data_paths], [tf.float32, tf.float32])
    mixture_mel.set_shape((None, None, 1))
    sources_mel.set_shape((None, None, 4))
    return mixture_mel, sources_mel

bucket_boundaries = [1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 30, 
                     31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 55, 
                     56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 
                     80, 81, 82, 85, 86, 87]
bucket_batch_sizes = [BATCH_SIZE] * (len(bucket_boundaries) + 1)

def load_data(data_paths, is_shuffle=False):
    def generator():
        for data_path in data_paths:
            yield data_path

    # Create a dataset from the file names
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(), 
        output_signature=(
            tf.TensorSpec(shape=(5,), dtype=tf.string)
        )
    )

    # Apply the function to the dataset
    dataset = dataset.map(tf_load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    def print_shapes(element1, element2):
        print('Shape of mixture_mel:', tf.shape(element1))
        print('Shape of sources_mel:', tf.shape(element2))
        return element1, element2


    # dataset = dataset.apply(
        # tf.data.experimental.bucket_by_sequence_length(
    dataset = dataset.bucket_by_sequence_length(
            element_length_func=element_length_func,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes
        )
    # dataset = dataset.map(print_shapes)
    # )
    if(is_shuffle):
        dataset = dataset.shuffle(BUFFER_SIZE)

    # Use prefetch to allow the dataset to asynchronously fetch batches while your model is training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Load the data from the JSON file
with open(train_url, 'r') as f:
    train_data_paths = json.load(f)
with open(test_url, 'r') as f:
    test_data_paths = json.load(f)

# Load the training dataset
train_dataset = load_data(train_data_paths)
test_dataset = load_data(test_data_paths)

def get_unique_shapes(dataset):
    unique_shapes_1 = set()
    unique_shapes_2 = set()

    for element in dataset:
        # Add the shape to the set of unique shapes
        unique_shapes_1.add(element[0].shape[0])
        unique_shapes_2.add(element[0].shape[1])

    return (unique_shapes_1, unique_shapes_2)

# Model instantiation and training
num_channels = 1  # replace with the number of channels in your task
num_sources = num_channels * 4  # replace with the number of sources in your task
input_shape = (num_channels, 128, 82)  # replace with the actual input shape
print("Loading the model")
# model = SourceSeparationModel(num_sources, num_channels, input_shape)
model = WaveUNet(num_sources=num_sources)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=optimizer,
              loss=loss_fn)

# Checkpointing
model_dir = 'models/tf_v3'
checkpoint_filepath = f'{model_dir}/tf_checkpoint/model_checkpoint_v3.tf'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=False)

# Training
print("Training the model")
train_history = model.fit(train_dataset, 
                          epochs=EPOCHS, 
                          steps_per_epoch=len(train_data_paths) // BATCH_SIZE,
                          validation_data=test_dataset, 
                          callbacks=[model_checkpoint_callback])

# Save the entire model to a HDF5 file
model.save(f'{model_dir}/model_v3.tf')

# Load the best weights
# model.load_weights(checkpoint_filepath)

model.summary()

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Save the plot
plt.savefig('assets/loss_plot.png')

plt.show()

# output = model(input)
# output = tf.cast(output, tf.float32)
# loss = loss_fn(target, output)
