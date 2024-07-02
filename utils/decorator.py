import time


class Decorator:

    def __init__(self):
        pass

    @staticmethod
    def get_time(f):
        def inner(*arg, **kwarg):
            s_time = time.time()
            res = f(*arg, **kwarg)
            e_time = time.time()

            return e_time - s_time

        return inner