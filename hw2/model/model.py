import tensorflow as tf
import tensorflow_model_optimization as tfmot
from dataset import Dataset


def get_model(
        dataset: Dataset,
        prune: bool,
        initial_sparsity,
        final_sparsity,
        begin_step,
        end_step):
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=begin_step,
        end_step=end_step
        )
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
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        to_prune=model,
        pruning_schedule=pruning_schedule
    )
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    return (model_for_pruning, callbacks) if prune else (model, [])
    