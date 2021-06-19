import asyncio
import os
import threading
import time
from typing import List

from aiohttp import ClientSession


async def download_site(session: ClientSession, url: str):
    async with session.get(url) as response:
        content = await response.text()
        print(f'Read {len(content)} from {url} '
              f'with process {os.getpid()} and thread {threading.get_ident()}')


async def download_all_sites(urls: List[str]):
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(download_site(session, url))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    sites = ['https://blick.ch',
             'https://20min.ch'] * 80
    start_time = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_all_sites(sites))
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")
