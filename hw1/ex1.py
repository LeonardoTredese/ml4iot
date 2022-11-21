import scipy.io.wavfile as wf
import sounddevice as sd
import tensorflow as tf
import argparse
import os
from time import time

def get_audio_from_numpy(indata):
 indata = tf.convert_to_tensor(indata, dtype=tf.float32)
 indata = 2*(indata + 32768) / (32767 + 32768) -1 
 indata = tf.squeeze(indata)
 return indata

def get_spectrogram(indata, samplerate, frame_length_in_s, frame_step_in_s):
    audio = get_audio_from_numpy(indata)
    frame_length = int(frame_length_in_s * samplerate)
    frame_step = int(frame_step_in_s * samplerate)
    stft = tf.signal.stft(
        audio,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length = frame_length
    )
    spectrogram = tf.abs(stft)
    return spectrogram

def is_silence(indata, samplerate, frame_length_in_s, dbFSthresh, duration_time):
    spectrogram = get_spectrogram(
        indata,
        samplerate,
        frame_length_in_s,
        frame_length_in_s
    )
    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_time:
        return 0
    else:
        return 1

def get_callback(samplerate):
    def store_in_file(indata, frames, callback_time, status):
        filename =f'{str(time())}.wav'
        if not is_silence(indata, samplerate, 16e-3, -120, 17e-3):
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
        
