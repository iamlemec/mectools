import json
import redis

class Clip:
    def __init__(self):
        self.__dict__['red'] = redis.Redis(host='localhost', port=6379, db=0)

    def __setattr__(self, key, value):
        self.red[key] = json.dumps(value)

    def __getattr__(self, key):
        return json.loads(self.red[key])
