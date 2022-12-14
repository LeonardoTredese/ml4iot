import os
import uuid
import redis
import sounddevice as sd
import zipfile as zf
import tensorflow as tf
import psutil as psu
from functools import partial
from time import time
from argparse import ArgumentParser

# Constants
SR = 16_000
VAD_FRAME_LEN = 16e-3
VAD_DB_THRESH = -120
VAD_DURATION = 17e-3
MODEL_PATH = "model6.tflite"
VUI_FRAME_LEN = 32e-3
VUI_FRAME_STEP = 16e-3
VUI_NUM_MEL_BINS = 40
VUI_LOWER_FREQUENCY = 20
VUI_UPPER_FREQUENCY = 8000
VUI_NUM_COEFFICIENTS = 16
LINEAR_TO_MEL_WEIGHT_MATRIX = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins = VUI_NUM_MEL_BINS,
    num_spectrogram_bins = int(VUI_FRAME_LEN * SR) // 2 + 1,
    sample_rate = SR,
    lower_edge_hertz = VUI_LOWER_FREQUENCY,
    upper_edge_hertz = VUI_UPPER_FREQUENCY
)

# Script Arguments
parser = ArgumentParser()
parser.add_argument('--device', type=int, required=True)
parser.add_argument('--host', type=str, required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--user', type=str, required=True)
parser.add_argument('--password', type=str, required=True)

# Redis battery monitor
class BatteryMonitor:
    def __init__(self, host: str, port: int, user: str, password: str) -> None:
        self._percent_series = f'{hex(uuid.getnode())}:battery'
        self._power_series = f'{hex(uuid.getnode())}:power'
        self._monitoring = False
        self._redis = redis.Redis(host=host, username=user, port=port, password=password)
        # create redis TS
        if self._redis.ping(): 
            try:
                self._redis.ts().create(key=self._percent_series)
                self._redis.ts().create(key=self._power_series)
            except redis.ResponseError:
                pass
        else:
            print("Cannot connect to redis instance")

    def start(self) -> None:
        self._monitoring = True

    def stop(self) -> None:
        self._monitoring = False

    def log_battery(self) -> None:
        if self._monitoring and self._redis.ping():
            battery = psu.sensors_battery()
            try:
                self._redis.ts().add(key=self._percent_series, timestamp=int(time()*1000), value=battery.percent)
                self._redis.ts().add(key=self._power_series, timestamp=int(time()*1000), value=int(battery.power_plugged))
            except redis.ResponseError:    
                print("Could not log")

# Utils
def get_audio_from_numpy(indata):
 indata = tf.convert_to_tensor(indata, dtype=tf.float32)
 indata = 2*(indata + 32768) / (32767 + 32768) -1 
 indata = tf.squeeze(indata)
 return indata

def get_log_mel_spectrogram(spectrogram):
    mel_spectrogram = spectrogram @ LINEAR_TO_MEL_WEIGHT_MATRIX
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram

def get_mfccs(log_mel_spectrogram):
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    return mfccs[:, :VUI_NUM_COEFFICIENTS]

def preprocess(indata):
    data = get_spectrogram(indata, SR, VUI_FRAME_LEN, VUI_FRAME_STEP)
    data = get_log_mel_spectrogram(data)
    return get_mfccs(data)
    
# VAD
def get_spectrogram(indata, samplerate, frame_length_in_s, frame_step_in_s):
    audio = get_audio_from_numpy(indata)
    frame_length = int(frame_length_in_s * samplerate)
    frame_step = int(frame_step_in_s * samplerate)
    stft = tf.signal.stft(
        audio,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length = frame_length
    )
    spectrogram = tf.abs(stft)
    return spectrogram

def is_silence(indata, samplerate, frame_length_in_s, dbFSthresh, duration_time):
    spectrogram = get_spectrogram(
        indata,
        samplerate,
        frame_length_in_s,
        frame_length_in_s
    )
    dbFS = 20 * tf.math.log(spectrogram + 1e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s
    return int(non_silence_duration <= duration_time)

# TFlite model wrapper
class Model:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def predict(self, sample):
        sample = tf.expand_dims(sample, -1)
        sample = tf.expand_dims(sample, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'], sample)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

### Core ###
def VAD_VUI(indata, frames, callback_time, status, model = None, monitor= None):
    if not is_silence(indata, SR, VAD_FRAME_LEN, VAD_DB_THRESH, VAD_DURATION):
        print("detected sound")
        x = preprocess(indata)
        y = model.predict(x)[0]
        if y[0] > .95:
            print("detected go")
            monitor.start()
        elif y[1] > .95:
            print("detected stop")
            monitor.stop()
    monitor.log_battery()

args = parser.parse_args()

if not os.path.exists(MODEL_PATH) and os.path.exists(MODEL_PATH + '.zip'):
    with zf.ZipFile(MODEL_PATH + '.zip', 'r') as zipper:
        print(zipper.infolist())
        zipper.extractall(path=os.getcwd())

if not os.path.exists(MODEL_PATH):
    print(f"Could not find either {MODEL_PATH} or {MODEL_PATH + '.zip'}, Exiting")
    exit(1)

monitor = BatteryMonitor(args.host, args.port, args.user, args.password)
model = Model(MODEL_PATH)
callback = partial(VAD_VUI, model=model, monitor=monitor)

with sd.InputStream(callback=callback, device=args.device, dtype='int16', samplerate=SR, channels=1, blocksize=SR):
    while command := input("commands: q = stop: "):
        if command in ['q', 'Q']:
            break
