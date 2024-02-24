import os
import torchaudio

def chunk_audio_files(root_dir, chunk_length_sec=10):
    # Define the subdirectories for each instrument
    instruments = ['mixture', 'bass', 'drums', 'vocals', 'other']

    # Iterate over the train and test directories
    for dataset in ['train', 'test']:
        dataset_dir = os.path.join(root_dir, dataset)

        # Iterate over each song directory
        for song_id, song_dir in enumerate(os.listdir(dataset_dir)):
            song_dir_path = os.path.join(dataset_dir, song_dir)
            if not os.path.isdir(song_dir_path):
                continue

            # Iterate over each instrument
            for instrument in instruments:
                audio_file_path = os.path.join(song_dir_path, f'{instrument}.wav')

                # Load the audio file
                waveform, sample_rate = torchaudio.load(audio_file_path)
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)(waveform)
                sample_rate = 44100

                # Calculate the number of frames per chunk
                frames_per_chunk = chunk_length_sec * sample_rate

                # Split the audio file into chunks and save them
                chunk_id = 0
                for i in range(0, waveform.size(1), frames_per_chunk):
                    chunk = waveform[:, i:i+frames_per_chunk]

                    # Create the output directory if it doesn't exist
                    output_dir = os.path.join('data', dataset, instrument)
                    os.makedirs(output_dir, exist_ok=True)

                    print(f'Saving chunk {chunk_id} of {instrument} from song {song_id} in {dataset} dataset in {output_dir}...')

                    # Save the chunk
                    output_file_path = os.path.join(output_dir, f'{song_id}_{chunk_id}.wav')
                    torchaudio.save(output_file_path, chunk, sample_rate)

                    chunk_id += 1

# Call the function
chunk_audio_files('./dataset')