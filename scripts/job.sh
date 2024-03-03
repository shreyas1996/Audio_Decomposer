SUFFIX=$(date +%s)
bsub -q hpc \
     -J AUDIO_DECOMPOSER_$SUFFIX\
     -n 8 \
     -gpu "num=1:mode=exclusive_process" \
     -W 24:00 \
     -R "select[gpu32gb] rusage[mem=32GB]" \
     -o /zhome/49/b/174072/audio_decomposer/outputs/AUDIO_DECOMPOSER_$SUFFIX_%J.out \
     -e /zhome/49/b/174072/audio_decomposer/outputs/AUDIO_DECOMPOSER_$SUFFIX_%J.err \
     "/zhome/49/b/174072/audio_decomposer/start_train.sh"