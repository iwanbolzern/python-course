from typing import Callable


def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


def calculate(operation: Callable[[int, int], int],
              a: int, b: int) -> int:
    """Demonstration of first class citizen"""
    return operation(a, b)


if __name__ == '__main__':
    # first class citizen
    print(f'1 + 1 = {calculate(add, 1, 1)}')
    print(f'1 - 1 = {calculate(subtract, 1, 1)}')
