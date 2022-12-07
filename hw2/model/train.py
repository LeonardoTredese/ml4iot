import tensorflow as tf
from model import get_model
from dataset import Dataset
from argparse import ArgumentParser
import os
import save
import parameters
import pandas as pd
import utils


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
    model, callbacks = get_model(
        dataset=dataset,
        prune=args.prune,
        initial_sparsity=args.initial_sparsity,
        final_sparsity=args.final_sparsity,
        begin_step=int(len(dataset.train_batch)*args.epochs*0.2),
        end_step=int(len(dataset.train_batch)*args.epochs)
        )
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.initial_learning_rate,
        end_learning_rate=args.end_learning_rate,
        decay_steps=len(dataset.train_batch) * args.epochs,
    )
    optimizer = tf.optimizers.Adam(learning_rate=linear_decay)
    metrics = [tf.metrics.SparseCategoricalAccuracy()]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(
        dataset.train_batch,
        epochs=args.epochs,
        validation_data=dataset.val_batch,
        callbacks=callbacks
        )
    latency_value = utils.compute_latency(model=model, dataset=dataset.test)
    test_loss, test_accuracy = model.evaluate(dataset.test_batch)
    training_loss = history.history['loss'][-1]
    training_accuracy = history.history['sparse_categorical_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]
    model_name = save.save(
        model=model,
        models_folder=args.models_folder
        )
    model_size, zip_size = save.convert_to_lite(
        models_folder=args.models_folder,
        model_name=model_name,
        tflite_models_folder=args.tflite_models_folder
        )
    model_info = vars(args)
    model_info['latency_ms'] = latency_value
    model_info['tflite_model_size_kB'] = model_size
    model_info['zip_tflite_model_size_kB'] = zip_size
    model_info['training_loss'] = training_loss
    model_info['training_accuracy'] = training_accuracy
    model_info['val_loss'] = val_loss
    model_info['val_accuracy'] = val_accuracy
    model_info['test_loss'] = test_loss
    model_info['test_accuracy'] = test_accuracy
    utils.print_pretty_results(model_info)
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
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parameters.IO(parser=parser)
    parameters.preprocess(parser=parser)
    parameters.pruning(parser=parser)
    parameters.training(parser=parser)
    args = parser.parse_args()
    utils.set_random_state(args.seed)
    utils.flush_folders(args)
    utils.create_folders(args)
    main(args)
