from functools import wraps

from constants import Log, VALID_SERIES_TYPES


def validateArgsTypes(logger, valid_types):
    def decorator(func):
        @wraps(func)
        def wrapper(x, *args, **kwargs):
            if type(x) not in valid_types:
                message = f"{Log.ERR} - {func.__name__} expected {valid_types} types but {type(x)} was passed!"
                logger.error(message)
                raise TypeError(message)
            return func(x, *args, **kwargs)
        return wrapper
    return decorator


def validate_vectors_with_same_length(func):
    """
    Decorator to validate the types and lengths of two vectors.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract x and y from positional or keyword arguments
        if 'x' in kwargs and 'y' in kwargs:
            x, y = kwargs['x'], kwargs['y']
        else:
            # Assume x and y are the first two positional arguments
            x, y = args[0], args[1]

        assert type(x) in VALID_SERIES_TYPES, f"{VALID_SERIES_TYPES} types were expected, but got {type(x)}"
        assert type(y) in VALID_SERIES_TYPES, f"{VALID_SERIES_TYPES} types were expected, but got {type(y)}"
        assert len(x) == len(y), f"Both vectors must have the same length, got {len(x)} and {len(y)}"

        return func(*args, **kwargs)
    return wrapper
