# 导入 functools 模块，提供高阶函数支持（如装饰器）
import functools
# 导入 time 模块，用于获取当前时间，计算函数运行时长
import time


def display_time(text):
    """
    装饰器工厂，用于计算并输出被装饰函数的运行时间。

    :param text: 个性化输出语句，用于说明输出的时间含义
    :return: 装饰器函数，该装饰器用于包装目标函数，输出运行时间信息
    """

    # 定义真正的装饰器函数，接受目标函数作为参数
    def decorator(func):
        # 使用 functools.wraps 保留原函数的信息（如函数名、docstring等）
        @functools.wraps(func)
        # 定义包装函数，接收任意位置参数和关键字参数
        def wrapper(*args, **kwargs):
            # 记录目标函数开始执行时的时间
            begin_time = time.time()
            # 执行目标函数，并保存返回结果
            result = func(*args, **kwargs)
            # 记录目标函数结束执行时的时间
            end_time = time.time()
            # 计算运行时间，并输出格式化后的结果
            print("{} {} time {:.4} seconds".format(func.__name__, text, end_time - begin_time))
            # 返回目标函数的执行结果
            return result

        # 返回包装后的函数
        return wrapper

    # 返回装饰器函数
    return decorator
