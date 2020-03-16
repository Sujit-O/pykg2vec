import logging
import threading

class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class Singleton(_Singleton("SingletonMeta", (object,), {})):
    pass


class Logger(Singleton):

    _FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _VERBOSE = False

    def __init__(self):
        self.loggers = {}
        self.lock = threading.Lock()

    def get_logger(self, name):
        self.lock.acquire()
        if self.loggers.get(name) is None:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG if Logger._VERBOSE else logging.INFO)
            formatter = logging.Formatter(Logger._FORMAT)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            if not logger.handlers:
                logger.addHandler(console_handler)

            self.loggers[name] = logger
        self.lock.release()

        return self.loggers.get(name)

