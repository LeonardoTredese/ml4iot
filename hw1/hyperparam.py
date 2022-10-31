import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
from tqdm import tqdm
from preprocessing import * 

dataset_dir = 'data'

def is_silence(filename, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres):
    spectrogram, samp_rate, label = get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_length_in_s)
    dbFS = 20 * tf.math.log(spectrogram + 1e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    silent_frames = tf.reduce_sum(tf.cast(energy < dbFSthres, tf.int8))
    return duration_thres < float(silent_frames) * frame_length_in_s
    
scatter_matrix = np.zeros((2,2))
start = time()
for filename in tqdm(os.listdir(dataset_dir)):
    ground_truth = filename.startswith('silence')
    filename = os.path.join(dataset_dir, filename)
    silent = is_silence(filename, 16000, .1, -121, .85)
    scatter_matrix[int(ground_truth), int(silent)] += 1
count = scatter_matrix.sum()
probs = scatter_matrix / count
acc = probs[0, 0] + probs[1, 1]
end = time()

count = scatter_matrix.sum()
print(f'delay:  {(end - start) / count:.3f}s')
probs = scatter_matrix / count
print(f'TN {probs[0, 0]:.2%}')
print(f'TP {probs[1, 1]:.2%}')
print(f'FN {probs[1, 0]:.2%}')
print(f'FP {probs[0, 1]:.2%}')
print(f'Accuracy  {probs[0, 0] + probs[1, 1]:.2%}')
