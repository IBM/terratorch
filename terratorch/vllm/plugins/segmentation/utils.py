from io import BytesIO
import aiohttp
from anyio import open_file

async def download_file_async(url: str) -> BytesIO:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise an error for bad responses
            return BytesIO(await response.read())
                
async def read_file_async(path: str) -> BytesIO:
    async with await open_file(path, "rb") as f:
        contents = await f.read()
        return BytesIO(contents)
