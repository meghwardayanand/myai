import math
import logging

from utils.common import validateArgsTypes
from constants import VALID_ACTIVATION_ARGS_TYPES


logger = logging.getLogger(__name__)


@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def sigmoid(x):
    if type(x) in [int, float]:
        return 1 / (1 + math.exp(-x))

    return [1/(1 + math.exp(-x_i)) for x_i in x]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def sigmoidPrime(x):
    result = sigmoid(x)
    if type(x) in [int, float]:
        return result * (1 - result)

    return [sig_i*(1 - sig_i) for sig_i in result]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def relu(x):
    if type(x) in [int, float]:
        return max(0, x)

    return [max(0, x_i) for x_i in x]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def reluPrime(x):
    if type(x) in [int, float]:
        return 1 if x > 0 else 0

    return [1 if x_i > 0 else 0 for x_i in x]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def softmax(x):
    maximum = max(x)
    exp_x = [math.exp(x_i) - maximum for x_i in x]
    summation = sum(exp_x)

    return [exp_x_i/summation for exp_x_i in exp_x]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def softmaxPrime(x):
    optimized_values = softmax(x)
    return [sm_i * (1 - sm_i) for sm_i in optimized_values]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def tanh(x):
    if type(x) in [int, float]:
        return math.tanh(x)

    return [math.tanh(x_i) for x_i in x]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def tanhPrime(x):
    if type(x) in [int, float]:
        return 1 - math.tanh(x) ** 2

    return [1 - math.tanh(x_i) ** 2 for x_i in x]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def leakyRelu(x, alpha=0.01):
    if type(x) in [int, float]:
        return x if x > 0 else alpha * x

    return [x_i if x_i > 0 else alpha * x_i for x_i in x]

@validateArgsTypes(logger, VALID_ACTIVATION_ARGS_TYPES)
def leakyReluPrime(x, alpha=0.01):
    if type(x) in [int, float]:
        return 1 if x > 0 else alpha

    return [1 if x_i > 0 else alpha for x_i in x]
