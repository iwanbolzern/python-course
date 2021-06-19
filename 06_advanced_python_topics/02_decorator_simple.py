def add(a: int, b: int) -> int:
    return a + b


def calc_decorator(func):
    def wrapper(*args, **kwargs):
        print('Something is happening before the calculation is performed.')
        res = func(*args, **kwargs)
        print('Something is happening after the calculation is performed.')
        return res

    return wrapper


if __name__ == '__main__':
    # decorated function (simple example)
    decorated_add = calc_decorator(add)
    print(f'1 + 1 = {decorated_add(1, 1)}')
