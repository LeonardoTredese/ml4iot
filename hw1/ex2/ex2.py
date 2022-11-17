import psutil
import time
import argparse
import uuid
import redis


class RedisDB:
    def __init__(self, host: str,
            port: int,
            user: str,
            password: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.redis_client = None
        self.connected = False

    def connect(self) -> bool:
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=self.host,
                username=self.user,
                port=self.port,
                password=self.password
                )
            self.connected = self.redis_client.ping()
            return self.connected
        return False

    def disconnect(self) -> bool:
        if self.redis_client is not None:
            self.redis_client.close()
            self.redis_client = None
            self.connected = False
            return True
        return False

    def is_connected(self) -> bool:
        if self.redis_client is not None:
            self.connected = self.redis_client.ping()
            return self.connected
        return False

    def create_ts(self, key: str,
            compression: bool=True,
            retention_msecs: int=0) -> bool:
        if self.is_connected():
            try:
                self.redis_client.ts().create(
                    key=key,
                    compression=compression,
                    retention_msecs=retention_msecs
                    )
                return True
            except redis.ResponseError:
                pass
        return False

    def delete_ts(self, key: str) -> bool:
        if self.is_connected():
            try:
                self.redis_client.delete(key)
                return True
            except redis.ResponseError:
                pass
        return False

    def add_value_ts(self, key: str,
            timestamp: int,
            value: int) -> bool:
        if self.is_connected():
            try:
                self.redis_client.ts().add(
                    key=key,
                    timestamp=timestamp,
                    value=value
                    )
                return True
            except redis.ResponseError:
                pass
        return False
    
    def create_rule_ts(self, source_key: str,
                dest_key: str,
                bucket_size_msec: int,
                aggregation_type: str) -> bool:
        if self.is_connected():
            try:
                self.redis_client.ts().createrule(
                    source_key=source_key,
                    dest_key=dest_key,
                    bucket_size_msec=bucket_size_msec,
                    aggregation_type=aggregation_type   
                    )
                return True
            except redis.ResponseError:
                pass
        return False


def main(args: dict) -> None:
    MAC_ADDRESS = hex(uuid.getnode())
    TS_BATTERY = f'{MAC_ADDRESS}:battery'
    TS_POWER = f'{MAC_ADDRESS}:power'
    TS_PLUGGED_SEC = f'{MAC_ADDRESS}:plugged_seconds'
    db = RedisDB(
        host=args.host, 
        port=args.port,
        user=args.user,
        password=args.password
        )
    print(db.connect())
    db.create_ts(
        key=TS_BATTERY,
        retention_msecs=int(2621440e4) # appr. 30 days, 5MB
        )
    db.create_ts(
        TS_POWER,
        retention_msecs=int(32768e5)
        )
    db.create_ts(
        TS_PLUGGED_SEC,
        retention_msecs=524288*24*60*60*1000 # appr. 1436 years, 1MB
        )
    db.create_rule_ts(
        source_key=TS_POWER,
        dest_key=TS_PLUGGED_SEC,
        bucket_size_msec=24*60*60*1000, # one record per day
        aggregation_type='sum'
        )
    try:
        print('Starting populating the DB, press CTRL+C to stop')
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
            print(f'Added values\nbattery: {value.percent}\npower:{int(value.power_plugged)}')
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nBye')


if __name__ == '__main__':
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
    main(parser.parse_args())
