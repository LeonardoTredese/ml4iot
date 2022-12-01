import tensorflow as tf
from dataset import Dataset
import argparse
import os


PREPROCESS_ARGS = {
    'batch_size': 10,
    'frame_length_in_s': 0.04, # 40 ms
    'frame_step_in_s': 0.02,   # overlap of 50%
    'num_mel_bins': 40,
    'lower_frequency': 20,
    'upper_frequency': 4000,
    'num_coefficients': 40
}

TRAINING_ARGS = {
    'initial_learning_rate': 0.01,
    'end_learning_rate': 1e-5,
    'epochs': 10
}

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
        os.path.join(ds_path,'msc-val','stop*'))
        )
    dataset = Dataset(
        train_files_ds=train_files_ds,
        test_files_ds=test_files_ds,
        val_files_ds=val_files_ds,
        preprocess='mfccs',
        **PREPROCESS_ARGS
    )
    model = get_model(dataset)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    initial_learning_rate = TRAINING_ARGS['initial_learning_rate']
    end_learning_rate = TRAINING_ARGS['end_learning_rate']
    epochs = TRAINING_ARGS['epochs']
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
        default='spect',
        help='Type of preprocess, str in \
            ["spect", "log_mel_spect", "mfccs"]'
    )
    main(parser.parse_args())
