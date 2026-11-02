# Document ID: fastapi-concurrency
# Title: Concurrency, Async/Await, and Threadpools
# Source: Python Documentation & FastAPI Guides (Python Software Foundation & Tiangolo, MIT License)
# Category: Concurrency Model

## Section 1: Async vs Sync Execution
In asynchronous programming with `async` and `await`, non-blocking I/O operations (such as database queries, external HTTP calls, and file reads) release the event loop, allowing the process to handle thousands of concurrent client connections simultaneously on a single CPU core.

## Section 2: When to Use async def vs def
If a path operation calls external asynchronous libraries using `await`, it must be declared with `async def`. If the code uses legacy synchronous blocking libraries (such as blocking database drivers or CPU-bound transformations), declaring standard `def` ensures FastAPI runs the function in an external worker threadpool rather than freezing the event loop.

## Section 3: CPU-bound Work vs I/O-bound Work
Async/await is optimized for I/O-bound workloads. For heavy CPU-bound computational workloads (such as image processing, machine learning model inference, or cryptography), running code in the main thread blocks all HTTP traffic. CPU-bound tasks should be delegated to multiprocessing worker pools or external job queues like Celery.
