import threading

thread_local = threading.local()


def hi():
    initialized = getattr(thread_local, 'initialized', None)
    if initialized is None:
        print("Nice to meet you", threading.current_thread().name)
        thread_local.initialized = True
    else:
        print("Welcome back", threading.current_thread().name)


def say_hi_n_times(n: int):
    for _ in range(n):
        hi()


if __name__ == '__main__':
    say_hi_n_times(2)

    thread = threading.Thread(target=say_hi_n_times, args=(2,))
    thread.start()

    thread = threading.Thread(target=say_hi_n_times, args=(2,))
    thread.start()
