# Document ID: fastapi-intro
# Title: FastAPI Overview and Core Architecture
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Framework Architecture

## Section 1: Introduction and Core Features
FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints. The key features of FastAPI include high execution speed on par with NodeJS and Go (thanks to Starlette and Pydantic), fast development cycles, reduced human-induced bugs by approximately 40%, and automatic interactive API documentation generated directly from OpenAPI schemas.

## Section 2: Under the Hood - Starlette and Pydantic
FastAPI is built on top of Starlette for the web routing and ASGI tooling, and Pydantic for data validation and schema serialization. Starlette provides asynchronous request handling, WebSockets, GraphQL support, and lifespan state management. Pydantic performs runtime type checking, coercion, and generates compliant JSON Schema definitions for every request body and response model.

## Section 3: Asynchronous Execution Model
FastAPI supports both standard synchronous functions (`def`) and asynchronous coroutines (`async def`). When a path operation is defined with `async def`, FastAPI executes it directly on the main event loop. If a standard `def` endpoint is declared, FastAPI automatically offloads execution to an external threadpool (using Starlette's `run_in_threadpool`) to prevent blocking the event loop during synchronous I/O operations.
