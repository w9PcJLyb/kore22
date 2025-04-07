import os
import uuid
import logging

FILE = "game.log"
IS_KAGGLE = os.path.exists("/kaggle_simulations")
LEVEL = logging.DEBUG if not IS_KAGGLE else logging.INFO
LOGGING_ENABLED = False
if IS_KAGGLE:
    LOGGING_ENABLED = True


class _FileHandler(logging.FileHandler):
    def __init__(self, *args, is_kaggle=False, **kwargs):
        self.is_kaggle = is_kaggle
        super().__init__(
            *args,
            **kwargs,
        )

    def emit(self, record):
        if self.is_kaggle:
            print(self.format(record))
        else:
            super().emit(record)


def init_logger(_logger):
    if not LOGGING_ENABLED:
        return

    if not IS_KAGGLE:
        if os.path.exists(FILE):
            os.remove(FILE)

    while _logger.hasHandlers():
        if not _logger.handlers:
            break
        _logger.removeHandler(_logger.handlers[0])

    _logger.setLevel(LEVEL)
    ch = _FileHandler(FILE, is_kaggle=IS_KAGGLE)
    ch.setLevel(LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H-%M-%S"
    )
    ch.setFormatter(formatter)
    _logger.addHandler(ch)


logger = logging.getLogger(uuid.uuid4().hex)
