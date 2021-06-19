import functools


def decorator_without_functools(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def decorator_with_functools(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@decorator_without_functools
def do_work_without(a: int, b: int) -> int:
    return a + b


@decorator_with_functools
def do_work_with(a: int, b: int) -> int:
    return a + b


if __name__ == '__main__':
    # who are you, really?
    print(f'Function name: {do_work_without.__name__}')
    print(f'Function name: {do_work_with.__name__}')
