import subprocess
import os
import librosa
import concurrent.futures
import soundfile as sf

def compress_and_upload_audio(file_path, output_path, bit_rate='256k'):
    try:
        print(f'Compressing {file_path} in {bit_rate} to {output_path}...')
        subprocess.run(['ffmpeg', '-i', file_path, '-ab', bit_rate, output_path])
        # subprocess.run(['ffmpeg', '-y', '-i', local_path, '-ar', str(new_sample_rate), '-ab', BIT_RATE, new_path])

        # Remove the temporary file
        os.remove(file_path)
        return "Done"
    except Exception as e:
        print(f'Error compressing {file_path} to {output_path}: {e}')
        return e

def process_instrument(song_id, dataset_dir, song_dir_path, instrument, destination, chunk_length_sec=1, bit_rate='256k'):
    audio_file_path = os.path.join(song_dir_path, f'{instrument}.wav')

    # Load the audio file
    waveform, sample_rate = librosa.load(audio_file_path, sr=None)
    if sample_rate != 44100:
        print(f'Interesting: {audio_file_path} has sample rate {sample_rate}')
        waveform = librosa.resample(waveform, sample_rate, 44100)
        sample_rate = 44100

    # Calculate the number of frames per chunk
    frames_per_chunk = chunk_length_sec * sample_rate

    # Split the audio file into chunks and save them
    chunk_id = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(0, waveform.size, frames_per_chunk):
            chunk = waveform[i:i+frames_per_chunk]

            # Create the output directory if it doesn't exist
            output_dir = os.path.join(destination, dataset_dir, instrument)
            os.makedirs(output_dir, exist_ok=True)

            print(f'Saving chunk {chunk_id} of {instrument} from song {song_id} in {dataset_dir} dataset in {output_dir}...')

            # Save the chunk
            tmp_file_path = os.path.join(output_dir, f'tmp_{song_id}_{chunk_id}.flac')
            # librosa.save(tmp_file_path, chunk, sample_rate)
            sf.write(tmp_file_path, chunk, sample_rate)
            output_file_path = os.path.join(output_dir, f'{song_id}_{chunk_id}.flac')

            # Compress and upload the chunk
            executor.submit(compress_and_upload_audio, tmp_file_path, output_file_path, bit_rate)

            chunk_id += 1

def chunk_audio_files(root_dir, destination, chunk_length_sec=1, bit_rate='256k'):
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        # Iterate over the train and test directories
        for dataset in ['train', 'test']:
            dataset_dir = os.path.join(root_dir, dataset)

            # Iterate over each song directory
            for song_id, song_dir in enumerate(os.listdir(dataset_dir)):
                song_dir_path = os.path.join(dataset_dir, song_dir)
                if not os.path.isdir(song_dir_path):
                    continue

                # Iterate over each instrument
                for instrument in INSTRUMENTS:
                #    executor.submit(process_instrument, song_id, dataset, song_dir_path, instrument, destination, chunk_length_sec, bit_rate)
                   process_instrument(song_id, dataset, song_dir_path, instrument, destination, chunk_length_sec, bit_rate)

def compress_chunked_flac_files(root_dir, bit_rate='256k'):
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for dataset in ['train', 'test']:
                dataset_dir = os.path.join(root_dir, dataset)
                for instrument in INSTRUMENTS:
                    song_dir_path = os.path.join(dataset_dir, instrument)
                    if not os.path.isdir(song_dir_path):
                        continue
                    for song in os.listdir(song_dir_path):
                        if not song.endswith('.flac'):
                            continue
                        song_path = os.path.join(song_dir_path, song)
                        new_song_path = song_path.replace("tmp_", "")
                        executor.submit(compress_and_upload_audio, song_path, new_song_path, bit_rate)
       

# Constants
MAX_WORKERS = 100
BIT_RATE = '256k'
BIT_RATE_LIST = ['128k', '192k', '256k', '320k']
SAMPLE_RATE = 44100
CHUNK_LENGTH_SEC = 1
# Define the subdirectories for each instrument
INSTRUMENTS = ['mixture', 'bass', 'drums', 'vocals', 'other']


os.makedirs(f'data/{BIT_RATE}/flac', exist_ok=True)

# chunk_audio_files('dataset', f'data/{BIT_RATE}/flac', chunk_length_sec=CHUNK_LENGTH_SEC, bit_rate='256k')
compress_chunked_flac_files('data/256k/flac', bit_rate='256k')