import uuid
import argparse
from redis_db import RedisDB
import sounddevice as sd
from vad import get_callback


def main(args):
    MAC_ADDRESS = hex(uuid.getnode())
    TS_BATTERY = f'{MAC_ADDRESS}:battery'
    SR = 16_000
    db = RedisDB(
        host=args.host, 
        port=args.port,
        user=args.user,
        password=args.password
        )
    db.connect()
    print('Is Redis connected? ', db.is_connected())
    db.create_ts(key=TS_BATTERY)
    with sd.InputStream(
            callback=get_callback(SR),
            device=args.device,
            dtype='int16',
            samplerate=SR,
            channels=1,
            blocksize=SR):
        while input("enter 'q' to exit the program") not in ['q', 'Q', 'quit']:
            pass
        print('Exited')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=int,
        required=True,
        help='Microphone device ID.'
    )
    parser.add_argument(
        '--host',
        type=str,
        required=True,
        help='Redis host.'
    )
    parser.add_argument(
        '--port',
        type=int,
        required=True,
        help='Redis port.'
    )
    parser.add_argument(
        '--user',
        type=str,
        required=True,
        help='Redis username.'
    )
    parser.add_argument(
        '--password',
        type=str,
        required=True,
        help='Redis password.'
    )
    main(parser.parse_args())
