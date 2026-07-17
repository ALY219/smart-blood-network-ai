import asyncio
import httpx
import time
from functools import wraps

# --- STEP 2: The Latency Tracker Decorator ---
def time_it(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__name__}] executed in {end - start:.4f} seconds")
        return result
    return wrapper

# --- STEP 3: Concurrency with Asyncio ---
@time_it
async def fetch_data(client, url):
    response = await client.get(url)
    return response.status_code

@time_it
async def main():
    urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
        "https://jsonplaceholder.typicode.com/posts/3"
    ]
    
    # httpx.AsyncClient handles connection pooling efficiently
    async with httpx.AsyncClient() as client:
        # Fire off all requests concurrently
        results = await asyncio.gather(*(fetch_data(client, url) for url in urls))
        
    print(f"\nFinal Status Codes: {results}")

# Start the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())