import tensorflow as tf
from dataset import Dataset
import argparse
import os
import random
import numpy as np
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


def get_model(dataset: Dataset):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=dataset.get_sample_shape()[1:]
            ),
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            strides=[2, 2],
            use_bias=False,
            padding='valid'
            ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=[3, 3],
            strides=[1, 1], 
            use_bias=False,
            padding='same'
            ),
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[1, 1],
            strides=[1, 1],   
            use_bias=False
            ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=[3, 3],
            strides=[1, 1],
            use_bias=False,
            padding='same'
            ),
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[1, 1],
            strides=[1, 1],
            use_bias=False
            ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            units=len(dataset.LABELS)
            ),
        tf.keras.layers.Softmax()
        ])
    return model

def main(args):
    ds_path = args.dataset
    train_files_ds = tf.data.Dataset.list_files((
        os.path.join(ds_path, 'msc-train', 'stop*'),
        os.path.join(ds_path, 'msc-train', 'go*'))
        )
    test_files_ds = tf.data.Dataset.list_files((
        os.path.join(ds_path,'msc-test','stop*'),
        os.path.join(ds_path, 'msc-test','go*'))
        )
    val_files_ds = tf.data.Dataset.list_files(
        (os.path.join(ds_path,'msc-val','stop*'),
        os.path.join(ds_path,'msc-val','go*'))
        )
    dataset = Dataset(
        train_files_ds=train_files_ds,
        test_files_ds=test_files_ds,
        val_files_ds=val_files_ds,
        preprocess=args.preprocess,
        batch_size=args.batch_size,
        frame_length_in_s=args.frame_length_in_s,
        frame_step_in_s=args.frame_step_in_s,
        num_mel_bins=args.num_mel_bins,
        lower_frequency=args.lower_frequency,
        upper_frequency=args.upper_frequency,
        num_coefficients=args.num_coefficients
    )
    model = get_model(dataset)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    initial_learning_rate = args.initial_learning_rate
    end_learning_rate = args.end_learning_rate
    epochs = args.epochs
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=end_learning_rate,
        decay_steps=len(dataset.train_ds) * epochs,
    )
    optimizer = tf.optimizers.Adam(learning_rate=linear_decay)
    metrics = [tf.metrics.SparseCategoricalAccuracy()]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(
        dataset.train_ds,
        epochs=epochs,
        validation_data=dataset.val_ds
        )
    test_loss, test_accuracy = model.evaluate(dataset.test_ds)
    training_loss = history.history['loss'][-1]
    training_accuracy = history.history['sparse_categorical_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]
    print(f'Training Loss: {training_loss:.4f}')
    print(f'Training Accuracy: {training_accuracy*100.:.2f}%')
    print()
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy*100.:.2f}%')
    print()
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy*100.:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='../msc-data',
        help='folder path for msc dataset'
    )
    parser.add_argument(
        '--preprocess',
        type=str,
        default='mfccs',
        help='Type of preprocess, str in \
            ["spect", "log_mel_spect", "mfccs"]'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='The size for the batches.'
    )
    parser.add_argument(
        '--initial_learning_rate',
        type=float,
        default=0.01,
        help='The initial learning rate value.'
    )
    parser.add_argument(
        '--end_learning_rate',
        type=float,
        default=1e-5,
        help='The final learning rate value.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='The number of epochs.'
    )
    parser.add_argument(
        '--frame_length_in_s',
        type=float,
        default=0.04,
        help='The frame length for STFT.'
    )
    parser.add_argument(
        '--frame_step_in_s',
        type=float,
        default=0.02,
        help='The step size for STFT.'
    )
    parser.add_argument(
        '--num_mel_bins',
        type=int,
        default=40,
        help='The number of bins for the mel spectrogram.'
    )
    parser.add_argument(
        '--lower_frequency',
        type=int,
        default=20,
        help='The lower frequency for the mel spectrogram.'
    )
    parser.add_argument(
        '--upper_frequency',
        type=int,
        default=4000,
        help='The upper frequency for the mel spectrogram.'
    )
    parser.add_argument(
        '--num_coefficients',
        type=int,
        default=40,
        help='The number of coefficients in mfccs.'
    )
    main(parser.parse_args())
