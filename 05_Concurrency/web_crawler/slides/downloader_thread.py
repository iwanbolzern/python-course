import os
import threading
from threading import Thread

import requests


class DownloaderThreaded(Thread):

    def __init__(self, url: str):
        super().__init__()

        self._url = url

    def run(self):
        response = requests.get(self._url)
        print(f'Read {len(response.content)} from {self._url} with process {os.getpid()} and thread {threading.get_ident()}')


if __name__ == '__main__':
    d1_t = DownloaderThreaded('https://blick.ch')
    d2_t = DownloaderThreaded('https://20min.ch')
    d1_t.start()
    d2_t.start()
    d1_t.join()
    d2_t.join()
