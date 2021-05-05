import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests
import threading
import time


thread_local = threading.local()


def get_session():
    if not hasattr(thread_local, 'session'):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url: str):
    session = get_session()
    with session.get(url) as response:
        print(f'Read {len(response.content)} from {url} '
              f'with process {os.getpid()} and thread {threading.get_ident()}')


def download_all_sites(urls: List[str]):
    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(download_site, urls)


if __name__ == '__main__':
    sites = ['https://blick.ch',
             'https://20min.ch'] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f'Downloaded {len(sites)} in {duration} seconds')