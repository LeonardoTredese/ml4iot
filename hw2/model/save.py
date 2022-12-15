import os
import tensorflow as tf
from time import time
import zipfile


def save(model, models_folder: str) -> str:
    """
    Save the model in the models_folder, using the
    timestamp as model name.
    IN:
        - model: a tensorflow model.
        - models_folder: the absolute path in which to save the model.
    OUT:
        - the name assigned to the model (i.e., the timestamp).
    """
    model_name = str(int(time()))
    model_path = os.path.join(models_folder, model_name)
    model.save(model_path)
    return model_name
    

def convert_to_lite(
        models_folder: str,
        model_name: str,
        tflite_models_folder: str) -> tuple[float, float]:
    """
    Generate .tflite and .zip from a saved model
    (with the save function in this module).
    IN:
        - models_folder: the folder in which the models are saved.
        - model_name: the name of the model to be converted.
    OUT:
        - the size of the .tflite file
        - the size of the .zip file
    """
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
        f.write(
            filename=tflite_model_path,
            arcname=tflite_model_name
            )
    tflite_model_size = os.path.getsize(tflite_model_path) / 1024.0
    zip_tflite_model_size = os.path.getsize(zip_tflite_model_path) / 1024.0
    return tflite_model_size, zip_tflite_model_size, tflite_model_path
