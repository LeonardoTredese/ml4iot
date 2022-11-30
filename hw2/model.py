import tensorflow as tf
from preprocessing import Preprocessor
import argparse
import os
from preprocessing import *


MFCCS_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': 0.04, # 40 ms
    'frame_step_in_s': 0.02,   # overlap of 50%
    'num_mel_bins': 40,
    'lower_frequency': 20,
    'upper_frequency': 4000,
    'num_coefficients': 40
}

TRAINING_ARGS = {
    'batch_size': 20,
    'initial_learning_rate': 0.01,
    'end_learning_rate': 1e-5,
    'epochs': 10
}

def main(args):
    ds_path = args.dataset
    train_files_ds = tf.data.Dataset.list_files(
        [os.path.join(ds_path, 'msc-train', 'stop*'),
        os.path.join(ds_path, 'msc-train', 'go*')]
        )
    test_files_ds = tf.data.Dataset.list_files(
        [os.path.join(ds_path,'msc-test','stop*'),
        os.path.join(ds_path, 'msc-test','go*')]
        )
    val_files_ds = tf.data.Dataset.list_files(
        [os.path.join(ds_path,'msc-val','stop*'),
        os.path.join(ds_path,'msc-val','stop*')]
        )
    preprocessor = Preprocessor(
        train_files_ds=train_files_ds,
        test_files_ds=test_files_ds,
        val_files_ds=val_files_ds,
        **MFCCS_ARGS
    )
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='../msc-data',
        help='folder path for msc dataset'
    )
    main(parser.parse_args())
