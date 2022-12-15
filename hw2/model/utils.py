import os
import shutil
import random
import numpy as np
from time import time
import tensorflow as tf
from dataset import Dataset
from tqdm import tqdm
from model_loader import TFLite_Model


def create_folders(args) -> None:
    """
    Creates all the folders passed
    as parameters for performing the
    experiments.
    """
    folders = (
        args.models_folder,
        args.tflite_models_folder,
        args.results_folder,
    )
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def flush_folders(args) -> None:
    """
    Before starting the experiment,
    all the past results are deleted.
    """
    if args.clean:
        folders = (
            args.models_folder,
            args.tflite_models_folder,
            args.results_folder,
        )
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)


def set_random_state(seed: int) -> None:
    """
    Set the random state for the experiment.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def compute_latency(tflite_model_path, dataset: Dataset) -> float:
    """
    Compute the latency of a given model on
    a given dataset. This function compute
    the median over the latency values of
    each prediction.
    """
    latency_preprocess = np.empty(
        shape=(len(dataset.test_files_ds),),
        dtype=float
        )
    latency_inference = np.empty(
        shape=(len(dataset.test_files_ds),),
        dtype=float
    )
    model: TFLite_Model = TFLite_Model(model_path=tflite_model_path)
    print('Starting latency evaluation...')
    for i, filename in tqdm(enumerate(dataset.test_files_ds)):
        audio, label = dataset.get_audio_and_label(filename)
        start_preprocess = time()
        sample, _ = dataset.preprocess(audio, label)
        end_preprocess = time()
        model.predict(sample)
        latency_inference[i] = time() - end_preprocess
        latency_preprocess[i] = end_preprocess - start_preprocess
    latency_total = latency_preprocess + latency_inference
    print('Done')
    latency_indexes = np.argsort(latency_total)
    if len(latency_indexes) % 2 == 0:
        median_indexes = latency_indexes[
            int(len(latency_indexes)/2)-1 : int(len(latency_indexes)/2)+1
            ]
    else:
        median_indexes = latency_indexes[int(len(latency_indexes)/2)]
    median_latency = np.mean(latency_total[median_indexes])
    median_latency_preprocess = np.mean(latency_preprocess[median_indexes])
    median_latency_inference = np.mean(latency_inference[median_indexes])
    return median_latency_preprocess*1000, \
            median_latency_inference*1000, \
            median_latency*1000
    


def print_pretty_results(results: dict) -> None:
    for key in results.keys():
        print(f'{key}: {results[key]}')
