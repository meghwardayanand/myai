import math
import logging

from utils.common import validateArgsTypes


logger = logging.getLogger(__name__)


class Optimizer:
    valid_types = [int, float, list]

    @staticmethod
    @validateArgsTypes(logger, valid_types)
    def optimize(x):
        raise NotImplementedError

    @staticmethod
    @validateArgsTypes(logger, valid_types)
    def prime(x):
        raise NotImplementedError


class Sigmoid(Optimizer):
    @staticmethod
    @validateArgsTypes(logger, Optimizer.valid_types)
    def optimize(x):
        if type(x) in [int, float]:
            return 1 / (1 + math.exp(-x))

        return [1/(1 + math.exp(-x_i)) for x_i in x]

    @staticmethod
    @validateArgsTypes(logger, Optimizer.valid_types)
    def prime(x):
        result = Sigmoid.optimize(x)
        if type(x) in [int, float]:
            return result * (1 - result)

        return [sig_i*(1 - sig_i) for sig_i in result]


class ReLU(Optimizer):
    @staticmethod
    @validateArgsTypes(logger, Optimizer.valid_types)
    def optimize(x):
        if type(x) in [int, float]:
            return max(0, x)

        return [max(0, x_i) for x_i in x]

    @staticmethod
    @validateArgsTypes(logger, Optimizer.valid_types)
    def prime(x):
        if type(x) in [int, float]:
            return 1 if x > 0 else 0

        return [1 if x_i > 0 else 0 for x_i in x]


class Softmax(Optimizer):
    @staticmethod
    @validateArgsTypes(logger, Optimizer.valid_types)
    def optimize(x):
        maximum = max(x)
        exp_x = [math.exp(x_i) - maximum for x_i in x]
        summation = sum(exp_x)

        return [exp_x_i/summation for exp_x_i in exp_x]

    @staticmethod
    @validateArgsTypes(logger, Optimizer.valid_types)
    def prime(x):
        optimized_values = Softmax.optimize(x)
        return [sm_i * (1 - sm_i) for sm_i in optimized_values]
