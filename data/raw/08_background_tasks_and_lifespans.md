# Document ID: fastapi-background-tasks
# Title: Background Tasks and Application Lifespan Events
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Concurrency & Lifecycle

## Section 1: Declaring Background Tasks
FastAPI includes `BackgroundTasks` to execute operations after returning an HTTP response to the client. This is suitable for tasks like sending email receipts, dispatching webhook notifications, or writing audit traces without delaying client response latency.

## Section 2: Differences with Celery and Distributed Queues
`BackgroundTasks` runs in-process inside the same ASGI worker process. For resource-intensive CPU computation or heavy scheduled batch jobs that require persistent retry queues across multiple server nodes, dedicated distributed task queues like Celery, Dramatiq, or ARQ backed by Redis or RabbitMQ are recommended over simple background tasks.

## Section 3: Modern Lifespan Event Handlers
Modern FastAPI applications manage startup and shutdown lifecycle events using an asynchronous context manager passed to `FastAPI(lifespan=lifespan)`. Code before the `yield` statement initializes connection pools, warm caches, or ML models on application startup, while code after `yield` gracefully terminates worker threads and closes socket connections on shutdown.
