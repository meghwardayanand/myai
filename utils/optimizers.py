import math


class Optimizer:
    @staticmethod
    def optimize(x):
        raise NotImplementedError

    @staticmethod
    def prime(x):
        raise NotImplementedError


class Sigmoid(Optimizer):
    @staticmethod
    def optimize(x):
        if type(x) in [int, float]:
            return 1 / (1 + math.exp(-x))

        return [1/(1 + math.exp(-item)) for item in x]

    @staticmethod
    def prime(x):
        result = Sigmoid.optimize(x)
        if type(x) in [int, float]:
            return result * (1 - result)

        return [i*(1 - i) for i in result]


class ReLU(Optimizer):
    @staticmethod
    def optimize(x):
        if type(x) in [int, float]:
            return max(0, x)

        return [max(0, i) for i in x]

    @staticmethod
    def prime(x):
        if type(x) in [int, float]:
            return 1 if x > 0 else 0
    
        return [1 if i > 0 else 0 for i in x]
