import os
import shutil
import random
import numpy as np
from time import time
import tensorflow as tf


def create_folders(args) -> None:
    folders = (
        args.models_folder,
        args.tflite_models_folder,
        args.results_folder,
    )
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def flush_folders(args) -> None:
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def compute_latency(model, dataset) -> float:
    values = np.empty(
        shape=(len(dataset),),
        dtype=float
        )
    i = 0
    for sample, _ in dataset:
        start = time()
        model.predict(sample)
        values[i] = time() - start
        i += 1
    return np.median(values)*1e3


def print_pretty_results(results: dict) -> None:
    for key in results.keys():
        print(f'{key}: {results[key]}')
