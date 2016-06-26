import pickle
import socket
from base64 import b64decode, b64encode

import uuid

from json import dumps, loads, JSONEncoder

import sys

import urllib.request
import urllib.parse

import time

running = True


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(obj)
        return {'_python_object': b64encode(pickle.dumps(obj)).decode('utf-8')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(b64decode(dct['_python_object'].encode('utf-8')))
    return dct


def check_server_for_work(id,uuid):
    try:
        id_args = {'id': id,
                'uuid':uuid}

        with urllib.request.urlopen('http://localhost:8080/getDna?{0}'.format(urllib.parse.urlencode(id_args))) as response:
            js = response.readall().decode('utf-8')
            if js != '':
                x = loads(js, object_hook=as_python_object)
                x.get_fitness()
                y = dumps(x, cls=PythonObjectEncoder)

                result_args = {'result': y}

                with urllib.request.urlopen(('http://localhost:8080/dnaResult?{0}'.format(urllib.parse.urlencode(result_args)))) as valid:
                    print(valid)
    except (ConnectionRefusedError, urllib.error.URLError) as e:
        print("No server found!")
    except:
        print(sys.exc_info()[0])


id = socket.gethostname()
uuid = uuid.uuid1()
while running:
    check_server_for_work(id, uuid)
    time.sleep(1)
