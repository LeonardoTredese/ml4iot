import scipy.io.wavfile as wf
import sounddevice as sd
import tensorflow as tf
import argparse
import os
from time import time


def get_audio_from_numpy(indata):
 indata = tf.convert_to_tensor(indata, dtype=tf.float32)
 indata = (indata + 32768) / (32767 + 32768)
 indata = tf.squeeze(indata)
 return indata

def get_spectrogram(indata, samplerate, frame_length_in_s, frame_step_in_s):
    audio = get_audio_from_numpy(indata)
    frame_length = int(frame_length_in_s * samplerate)
    frame_step = int(frame_length_in_s * samplerate)
    stft = tf.signal.stft(
        audio,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length = frame_length
    )
    spectrogram = tf.abs(stft)
    return spectrogram

def is_silence(indata, samplerate, frame_length_in_s, dbFSthres, duration_thres):
    spectrogram = get_spectrogram(indata, samplerate, frame_length_in_s, frame_length_in_s)
    dbFS = 20 * tf.math.log(spectrogram + 1e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    silent_frames = tf.reduce_sum(tf.cast(energy < dbFSthres, tf.int8))
    return duration_thres < float(silent_frames) * frame_length_in_s

def get_callback(samplerate):
    def store_in_file(indata, frames, callback_time, status):
        filename =f'{str(time())}.wav'
        if not is_silence(indata, samplerate, .1, -121, .85):
            wf.write(filename, samplerate, indata)
    return store_in_file



if __name__  == '__main__':
    SR = 16_000
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int)
    dev = parser.parse_args().device 
    with sd.InputStream(callback=get_callback(SR), device=dev, dtype='int16', samplerate=SR, channels=1, blocksize=SR):
        while input("enter 'q' to exit the program") not in ['q', 'Q', 'quit']:
            pass
        print('Exited')
        
