import tensorflow as tf
import tensorflow_io as tfio


class Preprocessor:

    def __init__(
            self,
            train_files_ds,
            test_files_ds,
            val_files_ds,
            downsampling_rate: int,
            frame_length_in_s: float,
            frame_step_in_s: float,
            num_mel_bins: int,
            lower_frequency: int,
            upper_frequency: int,
            num_coefficients: int
            ):
        self.train_files_ds = train_files_ds
        self.test_files_ds = test_files_ds
        self.val_files_ds = val_files_ds
        self.downsampling_rate = downsampling_rate
        self.frame_length_in_s = frame_length_in_s
        self.frame_step_in_s = frame_step_in_s
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        self.spectrogram_shape = None
        self.set_spectrogram_shape()
        self.train_spectrogram_ds = self.train_files_ds.map(self.get_spectrogram)
        self.test_spectrogram_ds = self.test_files_ds.map(self.get_spectrogram)
        self.val_spectrogram_ds = self.val_files_ds.map(self.get_spectrogram)
        self.train_log_mel_ds = self.train_files_ds.map(self.get_log_mel_spectrogram)
        self.test_log_mel_ds = self.test_files_ds.map(self.get_log_mel_spectrogram)
        self.val_log_mel_ds = self.val_files_ds.map(self.get_log_mel_spectrogram)
        self.train_mfccs_ds = self.train_files_ds.map(self.get_mfccs)
        self.test_mfccs_ds = self.test_files_ds.map(self.get_mfccs)
        self.val_mfccs_ds = self.val_files_ds.map(self.get_mfccs)
        
    def set_spectrogram_shape(self):
        tmp_ds = self.test_files_ds.map(self.get_spectrogram).cache()
        for spectrogram, _ in tmp_ds.take(1):
            self.spectrogram_shape = spectrogram.shape

    def get_audio_and_label(self, filename):
        audio_binary = tf.io.read_file(filename)
        audio, sampling_rate = tf.audio.decode_wav(audio_binary) 
        path_parts = tf.strings.split(filename, '/')
        path_end = path_parts[-1]
        file_parts = tf.strings.split(path_end, '_')
        label = file_parts[0]
        audio = tf.squeeze(audio)
        zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
        audio_padded = tf.concat([audio, zero_padding], axis=0)
        audio_padded.set_shape(self.downsampling_rate)
        return audio_padded, sampling_rate, label

    def get_spectrogram(self, filename):
        audio, _, label = self.get_audio_and_label(filename)
        frame_length = int(self.frame_length_in_s * self.downsampling_rate)
        frame_step = int(self.frame_step_in_s * self.downsampling_rate)        
        stft = tf.signal.stft(
            audio,
            frame_length = frame_length,
            frame_step = frame_step,
            fft_length = frame_length
        )
        spectrogram = tf.abs(stft)
        return spectrogram, label

    def get_log_mel_spectrogram(self, filename):
        spectrogram, label = self.get_spectrogram(filename)
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = self.num_mel_bins,
            num_spectrogram_bins = spectrogram.shape[1],
            sample_rate = self.downsampling_rate,
            lower_edge_hertz = self.lower_frequency,
            upper_edge_hertz = self.upper_frequency
        )
        mel_spectrogram = spectrogram @ linear_to_mel_weight_matrix
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram, label

    def get_mfccs(self, filename):
        log_mel_spectrogram, label = self.get_log_mel_spectrogram(filename)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        return mfccs[:,:self.num_coefficients], label
