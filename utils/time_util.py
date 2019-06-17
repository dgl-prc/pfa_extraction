import time
import datetime
import save_function

def sec2str(timestamp, format):
    '''
    :param timestamp: "%Y-%m-%d %H:%M:%S"
    :param formsat:
    :return:
    '''
    if int(timestamp) < 0:
        return ""
    return str(time.strftime(format, time.localtime(timestamp)))


def current_timestamp():
    timestamp = time.time()
    return sec2str(timestamp, "%Y-%m-%d %H:%M:%S")


class Time():
    def __init__(self):
        self.init_time = self.now()
        self.time_elapsed = '0'

    def now(self):
        return datetime.datetime.now()

    def time_counter(self):
        self.time_elapsed = self.now() - self.init_time
        return self.time_elapsed


if __name__ == "__main__":
    print current_timestamp()
    print datetime.datetime.now()
    time = Time()
    for i in range(100000):
        continue
    print time.time_counter()