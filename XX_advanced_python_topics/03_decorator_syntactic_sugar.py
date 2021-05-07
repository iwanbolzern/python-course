from datetime import datetime


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
    # syntactic sugar
    print(f'1 + 1 = {do_work(1, 1)}')
