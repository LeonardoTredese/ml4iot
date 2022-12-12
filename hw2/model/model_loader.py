import tensorflow as tf


class TFLite_Model:

    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def predict(self, sample):
        self.interpreter.set_tensor(self.input_details[0]['index'], sample)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])
