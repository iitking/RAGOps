# Document ID: fastapi-deployment
# Title: Production Deployment, Uvicorn, Gunicorn, and Docker
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Operations & DevOps

## Section 1: ASGI Servers - Uvicorn and Hypercorn
Python web applications conforming to the ASGI specification run on high-performance event loop web servers like Uvicorn (powered by uvloop and httptools) or Hypercorn (supporting HTTP/3). In development, Uvicorn is launched with `--reload` for hot code updates.

## Section 2: Gunicorn with Uvicorn Worker Processes
In bare-metal or virtual machine production environments, Gunicorn acts as a process manager to fork multiple worker processes, utilizing `uvicorn.workers.UvicornWorker`. A common formula for worker count is `(2 * CPU_cores) + 1` to ensure maximum concurrency across multi-core server processors.

## Section 3: Containerized Multi-stage Docker Builds
For cloud deployments (Kubernetes, AWS ECS, Google Cloud Run), multi-stage Docker builds separate build dependencies (compilers, pip caches) from the minimal runtime container image. Running as a non-root user and configuring an explicit container HEALTHCHECK probe enhances operational security and cluster resilience.
