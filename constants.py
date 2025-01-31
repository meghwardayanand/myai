from enum import Enum


VALID_ACTIVATION_ARGS_TYPES = [int, float, list]
VALID_NUMERIC_TYPES = [int, float]
VALID_SERIES_TYPES = [list]

class Log(Enum):
    INF = '[Info]'
    ERR = '[Error]'
    WAR = '[Warning]'
