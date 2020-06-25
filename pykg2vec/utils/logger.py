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

    def __init__(self):
        self._loggers = {}
        self._log_level = logging.INFO
        self._lock = threading.Lock()

    def get_logger(self, name):
        self._lock.acquire()
        if self._loggers.get(name) is None:
            logger = logging.getLogger(name)
            logger.setLevel(self._log_level)
            formatter = logging.Formatter(Logger._FORMAT)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            if not logger.handlers:
                logger.addHandler(console_handler)

            self._loggers[name] = logger
        self._lock.release()

        return self._loggers.get(name)

    @property
    def level(self):
        return self._log_level

    @level.setter
    def level(self, value):
        for name in self._loggers:
            self._loggers[name].setLevel(value)

