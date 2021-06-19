import os
import threading
from multiprocessing import Process

import requests


class DownloaderProcess(Process):

    def __init__(self, url: str):
        super().__init__()

        self._url = url

    def run(self):
        response = requests.get(self._url)
        print(f'Read {len(response.content)} from {self._url} with process {os.getpid()} '
              f'and thread {threading.get_ident()}')


if __name__ == '__main__':
    d1_t = DownloaderProcess('https://blick.ch')
    d2_t = DownloaderProcess('https://20min.ch')
    d1_t.start()
    d2_t.start()
    d1_t.join()
    d2_t.join()
