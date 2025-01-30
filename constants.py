from enum import Enum


VALID_ACTIVATION_ARGS_TYPES = [int, float, list]


class Log(Enum):
    INF = '[Info]'
    ERR = '[Error]'
    WAR = '[Warning]'
