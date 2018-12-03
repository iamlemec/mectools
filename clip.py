import json
import redis

class Clip:
    def __init__(self, host='localhost', port=6379):
        self.__dict__['redis'] = redis.Redis(host=host, port=port, db=0)

    def __setattr__(self, key, value):
        self.redis[key] = json.dumps(value)

    def __getattr__(self, key):
        return json.loads(self.redis[key])
