import time
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore, RLock


class Counter:

    def __init__(self):
        self.lock = RLock()
        self.count = 0

    def increment_by(self, n: int):
        for _ in range(n):
            with self.lock:
                # read value
                current_count = self.count
                # we simulate a long running operation and force the OS to reschedule
                time.sleep(0.5)
                # write value
                self.count = current_count + 1
                print(self.count)


if __name__ == '__main__':
    counter = Counter()

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(counter.increment_by, [100] * 10)

    assert counter.count == 100
