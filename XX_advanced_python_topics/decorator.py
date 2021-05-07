from datetime import datetime
from typing import Callable


def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


def calculate(operation: Callable[[int, int], int],
              a: int, b: int) -> int:
    """Demonstration of first class citizen"""
    return operation(a, b)


def calc_decorator(func):
    def wrapper(*args, **kwargs):
        print('Something is happening before the calculation is performed.')
        res = func(*args, **kwargs)
        print('Something is happening after the calculation is performed.')
        return res

    return wrapper


def not_during_the_night(func):
    def wrapper(*args, **kwargs):
        if 7 <= datetime.now().hour < 22:
            return func(*args, **kwargs)
        else:
            raise RuntimeError('Not allowed to work between 22pm and 07am')

    return wrapper


@not_during_the_night
def do_work(a: int, b: int) -> int:
    return a + b


if __name__ == '__main__':
    # first class citizen
    print(f'1 + 1 = {calculate(add, 1, 1)}')
    print(f'1 - 1 = {calculate(subtract, 1, 1)}')

    # decorated function (simple example)
    decorated_add = calc_decorator(add)
    print(f'1 + 1 = {decorated_add(1, 1)}')

    # syntactic sugar
    print(f'1 + 1 = {do_work(1, 1)}')
