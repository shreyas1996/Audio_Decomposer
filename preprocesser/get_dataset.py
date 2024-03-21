from collections import defaultdict
# from flask import Flask, jsonify, send_file
import os
import json

# app = Flask(__name__)
# url = 'https://staticfiledatasets.s3.ap-south-1.amazonaws.com/audio_decomposer'
url = ''

def get_files(root_dir, mode):
    files_dict = defaultdict(list)
    sub_dirs = ['mixture', 'bass', 'drums', 'vocals', 'other']
    for sub_dir in sub_dirs:
        path = os.path.join(root_dir, mode, sub_dir)
        for file in os.listdir(path):
            if file.endswith('.flac'):
                file_id = file.split('_')[-2:]
                # files_dict['_'.join(file_id)].append(url + '/' + path + '/' + file)
                files_dict['_'.join(file_id)].append(path + '/' + file)
    files_list = list(files_dict.values())
    file_name = os.path.join('assets', f'{mode}_files_list.json')
    print(f'Creating {file_name}...')
    with open(file_name, 'w') as f:
        json.dump(files_list, f)
    return files_list

# @app.route('/train')
def train_files():
    train_files = get_files('data/256k/flac/', 'train')
    # return jsonify(train_files)

# @app.route('/test')
def test_files():
    test_files = get_files('data/256k/flac/', 'test')
    # return jsonify(test_files)

# @app.route('/file/<path:file_path>')
def send_file_api(file_path):
    file_path = os.path.join(file_path)
    if os.path.exists(file_path):
        # return send_file(file_path, as_attachment=True)
        pass
    else:
        # return jsonify({'error': 'File not found'}), 404
        pass
if __name__ == '__main__':
    # app.run(debug=True, port=5002)
    train_files()
    test_files()