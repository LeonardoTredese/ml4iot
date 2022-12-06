import os
import tensorflow as tf
from time import time
import zipfile


def save(model, models_folder: str) -> str:
    timestamp = int(time())
    model_name = str(timestamp)
    model_path = os.path.join(models_folder, model_name)
    model.save(model_path)
    return model_name
    

def convert_to_lite(
        models_folder: str,
        model_name: str,
        tflite_models_folder: str) -> tuple[float, float]:
    model_path = os.path.join(models_folder, model_name)
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    tflite_model_name = f'{model_name}.tflite'
    tflite_model_path = os.path.join(
        tflite_models_folder,
        tflite_model_name
        )
    with open(tflite_model_path, 'wb') as fp:
        fp.write(tflite_model)
    zip_tflite_model_path = f'{tflite_model_path}.zip'
    with zipfile.ZipFile(
            zip_tflite_model_path,
            'w',
            compression=zipfile.ZIP_DEFLATED) as f:
        f.write(tflite_model_path)
    tflite_model_size = os.path.getsize(tflite_model_path) / 1024.0
    zip_tflite_model_size = os.path.getsize(zip_tflite_model_path) / 1024.0
    return tflite_model_size, zip_tflite_model_size
