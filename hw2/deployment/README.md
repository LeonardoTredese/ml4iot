# Getting microphone device ID

To use sounddevice, you need the library PortAudio.

In Ubuntu 22.04 you can run

```sudo apt-get install libportaudio2```

```sudo apt-get install libasound-dev```

To get the list of available microphone device,
run the python instruction

```sounddevice.query_devices()```

