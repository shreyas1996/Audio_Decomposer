nvidia-smi
#load the required modules
# module load cuda/12.1
# /appl/cuda/12.1.0/samples/bin/x86_64/linux/release/deviceQuery
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
module load tensorrt/8.6.1.6-cuda-11.X
# Load the Python environment
source /zhome/49/b/174072/audio_decomposer/shreyas/bin/activate
# Run the Python script
python3 /zhome/49/b/174072/audio_decomposer/trainer_tf.py