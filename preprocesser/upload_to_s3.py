import os
import boto3
import concurrent.futures
from collections import defaultdict
import json

def upload_to_s3(file_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, s3_key)

def load_files_and_upload_audio(dir_path, bucket_name, aws_s3_key):
    subfolders = ['mixture', 'bass', 'drums', 'vocals', 'other']
    print(f'Uploading files from {dir_path} to S3...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for subfolder in subfolders:
            print(f'Uploading files from {subfolder}...')
            folder_path = os.path.join(dir_path, subfolder)
            for filename in os.listdir(folder_path):
                if filename.endswith('.flac'):
                    file_path = os.path.join(folder_path, filename)
                    s3_key = f'audio_decomposer/{aws_s3_key}/{subfolder}/{filename}'
                    print(f'Uploading {file_path} to {bucket_name}/{s3_key}...')
                    executor.submit(upload_to_s3, file_path, bucket_name, s3_key)


def list_all_objects(bucket_name, prefix):
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')

    # Initialize the response
    response = {}

    # Initialize the list of objects
    objects = []
    print(f'Listing objects in {bucket_name}/{prefix}...')
    # While there are more objects to be fetched
    while response.get('IsTruncated', False) or 'Contents' not in response:
        # Get the continuation token, if there is one
        continuation_token = response.get('NextContinuationToken', None)

        # List the objects
        if continuation_token:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token)
        else:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        # Add the objects to the list
        objects.extend(response['Contents'])

    return objects

def get_comined_file_list(prefix, object_list):
    files_dict = defaultdict(list)
   
    for url in object_list:
        file = url.split('/')[-1]
        if file.endswith('.wav'):
            file_id = file.split('_')[-2:]
            files_dict['_'.join(file_id)].append('https://staticfiledatasets.s3.ap-south-1.amazonaws.com/' + url)
    files_list = list(files_dict.values())
    file_name = os.path.join(f'{prefix}_files_list.json')
    with open(file_name, 'w') as f:
        json.dump(files_list, f)
    return 'Done'



# Upload audio files to S3 in the train and test directories
BUCKET_NAME = 'staticfiledatasets'
MAX_WORKERS = 100

train_dir = 'data/256k/flac/train'
test_dir = 'data/256k/flac/test'

load_files_and_upload_audio(train_dir, BUCKET_NAME, 'flac/train')
load_files_and_upload_audio(test_dir, BUCKET_NAME, 'flac/test')
