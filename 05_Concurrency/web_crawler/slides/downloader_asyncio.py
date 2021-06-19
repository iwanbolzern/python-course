import asyncio
import os
import threading

import aiohttp


class DownloaderAsyncIO:

    def __init__(self, url: str):
        self._url = url

    async def download_async(self):
        session = aiohttp.ClientSession()

        async with session.get(self._url) as response:
            content = await response.text()
            print(f'Read {len(content)} from {self._url} with process {os.getpid()} and thread {threading.get_ident()}')

        await session.close()


if __name__ == '__main__':
    d1_a = DownloaderAsyncIO('https://blick.ch')
    d2_a = DownloaderAsyncIO('https://20min.ch')

    loop = asyncio.get_event_loop()
    task_group = asyncio.gather(d1_a.download_async(),
                                d2_a.download_async())
    loop.run_until_complete(task_group)
