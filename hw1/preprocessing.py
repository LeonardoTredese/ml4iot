import tensorflow as tf
import tensorflow_io as tfio
import os 

device_spec = tf.DeviceSpec(job ="localhost", replica = 0, device_type = "CPU")

LABELS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

def get_audio_and_label(filename):
    bin_audio = tf.io.read_file(filename)
    audio, sample_rate = tf.audio.decode_wav(bin_audio)
    head, tail = os.path.split(filename)
    label = tail.split('_')[0]
    audio = tf.squeeze(audio)
    zero_padding = tf.zeros(sample_rate - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], axis=0)
    return audio, sample_rate, label   

def get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s):
    audio, rate, label = get_audio_and_label(filename)
    
    if downsampling_rate != rate:
        audio = tfio.audio.resample(audio, rate.numpy(), downsampling_rate)
    
    frame_length = int(frame_length_in_s * downsampling_rate)
    frame_step = int(frame_length_in_s * downsampling_rate)
    
    stft = tf.signal.stft(
        audio,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length = frame_length
    )
    spectrogram = tf.abs(stft)
    return spectrogram, downsampling_rate, label


def get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency):
    spectrogram, rate, label = get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s)
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins = num_mel_bins,
        num_spectrogram_bins = spectrogram.shape[1],
        sample_rate = rate,
        lower_edge_hertz = lower_frequency,
        upper_edge_hertz = upper_frequency
    )
    mel_spectrogram = spectrogram @ linear_to_mel_weight_matrix
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram, label


def get_mfccs(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_coefficients):
    log_mel_spectrogram, label = get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    return mfccs[:,:num_coefficients], label
