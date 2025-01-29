from functools import wraps

from constants import Log


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
