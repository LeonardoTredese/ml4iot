import redis


class RedisDB:
    def __init__(self, host: str,
            port: int,
            user: str,
            password: str) -> None:
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
        else:
            return False

    def disconnect(self) -> bool:
        if self.redis_client is not None:
            self.redis_client.close()
            self.redis_client = None
            self.connected = False
            return True
        else:
            return False

    def is_connected(self) -> bool:
        if self.redis_client is not None:
            self.connected = self.redis_client.ping()
            return self.connected
        else:
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
                    value=value)
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
