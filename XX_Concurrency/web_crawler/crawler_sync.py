import os
import threading
import time
from typing import List

import requests
from requests import Session


def download_site(url: str, session: Session):
    with session.get(url) as response:
        print(f'Read {len(response.content)} from {url} '
              f'with process {os.getpid()} and thread {threading.get_ident()}')


def download_all_sites(urls: List[str]):
    with requests.Session() as session:
        for url in urls:
            download_site(url, session)


if __name__ == '__main__':
    sites = ['https://blick.ch',
             'https://20min.ch'] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f'Downloaded {len(sites)} sites in {duration} seconds')
