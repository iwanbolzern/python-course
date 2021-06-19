from contextlib import contextmanager


class File:
    def __init__(self, file_name: str, method: str):
        self._file_name = file_name
        self._method = method
        self._file_obj = None

    def __enter__(self):
        # Code to acquire resource, e.g.:
        self._file_obj = open(self._file_name, self._method)
        return self._file_obj

    def __exit__(self, type, value, traceback):
        # Code to release resource, e.g.:
        self._file_obj.close()
        return True


@contextmanager
def custom_open(file_name: str, method: str):
    # Code to acquire resource, e.g.:
    file = open(file_name, method)
    try:
        yield file
    finally:
        # Code to release resource, e.g.:
        file.close()


if __name__ == '__main__':
    with File(__file__, 'r') as myself:
        print(myself.read())

    with File(__file__, 'r') as _:
        raise Exception()

    with custom_open(__file__, 'r') as myself:
        print(myself.read())
