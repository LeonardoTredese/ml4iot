import psutil
import time
import argparse
from redis_db import RedisDB
import configparser
import sys


def main(args: dict):
    MAC_ADDRESS = psutil.net_if_addrs()['wlp3s0'][2].address
    TS_BATTERY = f'{MAC_ADDRESS}:battery'
    TS_POWER = f'{MAC_ADDRESS}:power'
    TS_PLUGGED_SEC = f'{MAC_ADDRESS}:plugged_seconds'
    db = RedisDB(
        host=args['host'], 
        port=args['port'],
        user=args['user'],
        password=args['password']
    )
    print(db.connect())
    db.create_ts(TS_BATTERY)
    db.create_ts(TS_POWER)
    db.create_ts(TS_PLUGGED_SEC)
    db.create_rule_ts(
        source_key=TS_POWER,
        dest_key=TS_PLUGGED_SEC,
        bucket_size_msec=24*60*60*1e3,
        aggregation_type='sum'
        )
    try:
        print('starting populating the DB, press CTRL+C to stop')
        while True:
            value = psutil.sensors_battery()
            timestamp = int(time.time()*1000)
            db.add_value_ts(
                key=TS_BATTERY,
                timestamp=timestamp,
                value = value.percent
                )
            db.add_value_ts(
                key=TS_POWER,
                timestamp=timestamp,
                value = int(value.power_plugged)
            )
            print(f'Added values\nbattery: \
                {value.percent}\npower:{int(value.power_plugged)}')
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nBye')


def argParse() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host',
        type=str,
        help='The Redis Cloud host',
        )
    parser.add_argument(
        '--port',
        type=int,
        help='The Redis Cloud port',
        )
    parser.add_argument(
        '--user',
        type=str,
        help='The Redis Cloud user',
        )
    parser.add_argument(
        '--password',
        type=str,
        help='The Redis Cloud password',
        )
    return parser.parse_args()


def fileParse() -> dict:
    parser = configparser.ConfigParser()
    parser.read(sys.argv[1])
    return parser['db-config']


if __name__ == '__main__':
    main(argParse() if sys.argv[1].startswith('-') \
         else fileParse())
