import tensorflow as tf


class Dataset:

    def __init__(
            self,
            train_files_ds,
            test_files_ds,
            val_files_ds,
            preprocess: str,
            batch_size: int,
            frame_length_in_s: float,
            frame_step_in_s: float,
            num_mel_bins: int,
            lower_frequency: int,
            upper_frequency: int,
            num_coefficients: int
            ):
        """
        This class manages the dataset and its preprocessing.
        IN:
            - train_files_ds: a string tensorflow Dataset, in which
                              each object is the filename of a sample
                              in the training set.
            - test_file_ds: same as the previous but for the test set.
            - val_files_ds: same as the previous but for validation set.
            - preprocess: a string in ['mfccs', 'log_mel_spect', 'spect'],
                          i.e., the type of preprocessing to apply.
            - batch_size: the batch size
            - frame_length_in_s: parameter to compute the spectrogram.
            - frame_step_in_s: parameter to compute the spectrogram.
            - num_mel_bins: parameter to compute log mel spectrogram.
            - lower_frequency: parameter to compute log mel spectrogram.
            - upper_frequency: parameter to compute log mel spectrogram.
            - num_coefficients: parameter to compute mfccs.
        OUT:
            - train_ds: training dataset with the desired preprocessing,
                           ready to be iterated.
            - test_ds: same as previous but for the test dataset.
            - val_ds: same as previous but for the validation dataset.
        """
        self.train_files_ds = train_files_ds
        self.test_files_ds = test_files_ds
        self.val_files_ds = val_files_ds
        self.DOWNSAMPLING_RATE = 16000
        self.LABELS = ('go', 'stop',)
        self.batch_size = batch_size
        self.batch_sample_shape = None
        self.sample_shape = None
        self.spectrogram_sample_shape = None
        self.frame_length_in_s = frame_length_in_s
        self.frame_step_in_s = frame_step_in_s
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
    
        if preprocess in ['log_mel_spect', 'mfccs']:
            fft_length = int(self.frame_length_in_s * self.DOWNSAMPLING_RATE)
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins = self.num_mel_bins,
                num_spectrogram_bins = fft_length // 2 + 1,
                sample_rate = self.DOWNSAMPLING_RATE,
                lower_edge_hertz = self.lower_frequency,
                upper_edge_hertz = self.upper_frequency)
        else:
            self.linear_to_mel_weight_matrix = None
        
        if preprocess == 'spect':
            self.preprocess = lambda audio, label: \
                self.get_spectrogram(audio, label)
        elif preprocess == 'log_mel_spect':
            self.preprocess = lambda audio, label: \
                self.get_log_mel_spectrogram(
                    *self.get_spectrogram(audio, label)
                    )
        elif preprocess == 'mfccs':
            self.preprocess = lambda audio, label: \
                self.get_mfccs(
                    *self.get_log_mel_spectrogram(
                        *self.get_spectrogram(audio, label)
                        )
                    )
        else:
            raise Exception(f'{preprocess} preprocess is not supported.')
        self.train_wav_ds = self.train_files_ds.map(self.get_audio_and_label)
        self.test_wav_ds = self.test_files_ds.map(self.get_audio_and_label)
        self.val_wav_ds = self.val_files_ds.map(self.get_audio_and_label)
        self.train = self.train_wav_ds.map(self.preprocess)
        self.test = self.test_wav_ds.map(self.preprocess)
        self.val = self.val_wav_ds.map(self.preprocess)
        self.train_batch = self.train.batch(self.batch_size)
        self.test_batch = self.test.batch(self.batch_size)
        self.val_batch = self.val.batch(self.batch_size)
        

    def get_sample_batch_shape(self):
        if self.batch_sample_shape is None:
            for batch, _ in self.train_batch.take(1):
                pass
            self.batch_sample_shape = batch.shape + (1,)
        return self.batch_sample_shape

    def get_audio_and_label(self, filename):
        audio_binary = tf.io.read_file(filename)
        audio, sampling_rate = tf.audio.decode_wav(audio_binary) 
        path_parts = tf.strings.split(filename, '/')
        path_end = path_parts[-1]
        file_parts = tf.strings.split(path_end, '_')
        label = file_parts[0]
        audio = tf.squeeze(audio)
        zero_padding = tf.zeros(
            sampling_rate - tf.shape(audio),
            dtype=tf.float32
            )
        audio_padded = tf.concat(
            [audio, zero_padding],
            axis=0
            )
        audio_padded.set_shape(self.DOWNSAMPLING_RATE)
        label_id = tf.argmax(label==self.LABELS)
        return audio_padded, label_id

    def get_spectrogram(self, audio, label):
        frame_length = int(self.frame_length_in_s * self.DOWNSAMPLING_RATE)
        frame_step = int(self.frame_step_in_s * self.DOWNSAMPLING_RATE)        
        stft = tf.signal.stft(
            audio,
            frame_length = frame_length,
            frame_step = frame_step,
            fft_length = frame_length
        )
        spectrogram = tf.abs(stft)
        return spectrogram, label

    def get_log_mel_spectrogram(self, spectrogram, label):
        mel_spectrogram = spectrogram @ self.linear_to_mel_weight_matrix
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram, label

    def get_mfccs(self, log_mel_spectrogram, label):
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        return mfccs[:, :self.num_coefficients], label
