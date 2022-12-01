import tensorflow as tf
from dataset import Dataset
import argparse
import os
import random
import numpy as np
from time import time
import pandas as pd
from zipfile import ZipFile
import shutil
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


def get_all_file_paths(directory: str) -> list:
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


def compute_size(
        models_folder: str,
        model_name: int
        ) -> tuple[int, int]:
    model_path = os.path.join(models_folder, model_name)
    model_size = 0
    model_files = get_all_file_paths(model_path)
    zip_path = f'{model_path}.zip'
    with ZipFile(zip_path, 'w') as f:
        for file_path in model_files:
            f.write(file_path)
            model_size += os.path.getsize(file_path)
    zip_size = os.path.getsize(zip_path)
    return model_size, zip_size


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
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.initial_learning_rate,
        end_learning_rate=args.end_learning_rate,
        decay_steps=len(dataset.train_ds) * args.epochs,
    )
    optimizer = tf.optimizers.Adam(learning_rate=linear_decay)
    metrics = [tf.metrics.SparseCategoricalAccuracy()]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(
        dataset.train_ds,
        epochs=args.epochs,
        validation_data=dataset.val_ds
        )
    test_loss, test_accuracy = model.evaluate(dataset.test_ds)
    training_loss = history.history['loss'][-1]
    training_accuracy = history.history['sparse_categorical_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]
    timestamp = int(time())
    model_name = str(timestamp)
    model_path = os.path.join(args.models_folder, model_name)
    model.save(model_path)
    model_size, zip_size = compute_size(
        models_folder=args.models_folder,
        model_name=str(timestamp)
        )
    model_info = vars(args)
    model_info['model_size'] = model_size
    model_info['zip_size'] = zip_size
    model_info['training_loss'] = training_loss
    model_info['training_accuracy'] = training_accuracy
    model_info['val_loss'] = val_loss
    model_info['val_accuracy'] = val_accuracy
    model_info['test_loss'] = test_loss
    model_info['test_accuracy'] = test_accuracy
    df = pd.DataFrame(model_info, index=[0])
    output_path = os.path.join(
        args.results_folder,
        f'{model_name}.csv'
        )
    df.to_csv(
        output_path,
        mode='a',
        header=not os.path.exists(output_path),
        index=False
        )
    shutil.rmtree(args.models_folder)

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
    parser.add_argument(
        '--models_folder',
        type=str,
        default='models',
        help='Folder for saving the models and results.'
    )
    parser.add_argument(
        '--results_folder',
        type=str,
        default='results',
        help='Folder for saving csv results.'
    )
    main(parser.parse_args())
