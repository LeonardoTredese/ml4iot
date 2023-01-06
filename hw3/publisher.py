import paho.mqtt.client as mqtt
import uuid 
import json
import time
import psutil as psu

TOPIC = 's302294'
client = mqtt.Client()
client.connect('mqtt.eclipseprojects.io', 1883)


def battery_message() -> dict:
    battery = psu.sensors_battery()
    return {
        'mac_address': hex(uuid.getnode()),
        'timestamp': round(time.time() * 1000),
        'battery_level': battery.percent,
        'power_plugged': battery.power_plugged
    }

while True:
    msg_dict = battery_message()
    print(msg_dict)
    client.publish(TOPIC, json.dumps(msg_dict))
    time.sleep(1)
    
