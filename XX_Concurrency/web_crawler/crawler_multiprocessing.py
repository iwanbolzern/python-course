import multiprocessing
import os
import threading
import time
from typing import List

import requests

session = None


def get_session():
    global session
    if not session:
        session = requests.Session()
    return session


def download_site(url: str):
    session = get_session()
    with session.get(url) as response:
        print(f'Read {len(response.content)} from {url} '
              f'with process {os.getpid()} and thread {threading.get_ident()}')


def download_all_sites(urls: List[str]):
    with multiprocessing.Pool(processes=30) as pool:
        pool.map(download_site, urls)


if __name__ == "__main__":
    sites = ['https://blick.ch',
             'https://20min.ch'] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")
