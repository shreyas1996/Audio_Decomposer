import os
import time
import psutil

def get_memory_usage(pid):
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # return memory usage in MB

while True:
    with open('pid.txt', 'r') as f:
        pid = int(f.read())
    print(f"Memory usage of process {pid}: {get_memory_usage(pid)} MB")
    time.sleep(0.2)  # sleep for 1 second