import functools
import time


def display_time(text):
    """
    装饰器，用来计算并输出函数运行时间
    :param text: 个性化输出语句
    :return: 装饰器函数
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            begin_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print("{} {} time {:.4} seconds".format(func.__name__, text, end_time - begin_time))
            return result

        return wrapper

    return decorator
