import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=False, level="INFO",
           format="{time:HH:mm:ss} | {level} | {name}:{line} — {message}")
