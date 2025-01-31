from constants import VALID_NUMERIC_TYPES, VALID_SERIES_TYPES
from utils.common import validate_vectors_with_same_length


def mean(x):
    assert type(x) in VALID_SERIES_TYPES

    return sum(x)/len(x)

def variance(x, mu):
    assert type(mu) in VALID_NUMERIC_TYPES
    assert type(x) in VALID_SERIES_TYPES

    return sum([(x_i - mu)**2 for x_i in x]) / len(x)

@validate_vectors_with_same_length
def covariance(x, x_mu, y, y_mu):
    assert type(x_mu) in VALID_NUMERIC_TYPES
    assert type(y_mu) in VALID_NUMERIC_TYPES

    return sum([(x[i] - x_mu)*(y[i] - y_mu) for i in range(len(x))])

@validate_vectors_with_same_length
def dot(x, y):
    return sum([x[i]*y[i] for i in range(len(x))])

@validate_vectors_with_same_length
def modulus(x, y):
    return sum([x[i]*y[i] for i in range(len(x))]) ** (1/2)

@validate_vectors_with_same_length
def cos(x, y):
    return dot(x, y) / modulus(x, y)

def absolute(x):
    assert type(x) in VALID_NUMERIC_TYPES

    return x if x >= 0 else -1*x

def power(x, n):
    assert type(x) in VALID_NUMERIC_TYPES
    assert type(n) == int

    if n == 0:
        return 1

    half = power(x, n//2)
    if n % 2 == 0:
        return  half * half
    else:
        return half * half * x

@validate_vectors_with_same_length
def distance(x, y, r=2):
    assert type(r) == int

    return sum([power(absolute(x[i] - y[i]), r) for i in range(len(x))]) ** 1/r

@validate_vectors_with_same_length
def manhatton(x, y):
    return distance(x, y, r=1)

@validate_vectors_with_same_length
def euclidean(x, y):
    return distance(x, y, r=2)

@validate_vectors_with_same_length
def calculate_binary_matchings(x, y):
    M00 = 0
    M01 = 0
    M10 = 0
    M11 = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            if x[i] == 1:
                M11 += 1
            else:
                M00 += 1
        else:
            if x[i] == 0:
                M01 += 1
            else:
                M10 += 1

    return M00, M01, M10, M11

@validate_vectors_with_same_length
def simple_matching(x, y):
    """
        SMC(x, y) = no. of matches / no. of attributes
    """
    M00, M01, M10, M11 = calculate_binary_matchings(x, y)
    return (M00 + M11)/(M00 + M01 + M10 + M11)

@validate_vectors_with_same_length
def jaccard(x, y):
    """
        J(x, y) = no. of 11 matches / no. of non-both-zero attributes
    """
    _, M01, M10, M11 = calculate_binary_matchings(x, y)
    return M11/(M01 + M10 + M11)

@validate_vectors_with_same_length
def tanimoto(x, y):
    """
        Tanimoto --> Extended Jaccard Coefficient
    """
    x_dot_y = dot(x, y)
    return x_dot_y / (modulus(x, x) + modulus(y, y) - x_dot_y)
