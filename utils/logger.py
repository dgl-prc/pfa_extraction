import sys
import os

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass