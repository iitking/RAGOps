# Document ID: fastapi-middleware
# Title: Middleware Architecture and CORS Configuration
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Networking & HTTP

## Section 1: Custom ASGI Middleware
A middleware is a function that runs before every request is processed by a path operation and after every response is generated. Developers create custom middleware with `@app.middleware("http")` to calculate request execution latency, inject custom tracing headers (such as `X-Request-ID`), or log request payloads.

## Section 2: Cross-Origin Resource Sharing (CORS)
CORS is a browser security mechanism that restricts web applications running at one origin from requesting resources from a different origin. FastAPI integrates Starlette's `CORSMiddleware` to authorize cross-origin requests by configuring `allow_origins`, `allow_credentials`, `allow_methods`, and `allow_headers`.

## Section 3: GZip and TrustedHost Middleware
FastAPI provides pre-built middleware components including `GZipMiddleware` to compress HTTP response payloads exceeding a minimum byte threshold (typically 1,000 bytes) and `TrustedHostMiddleware` to protect applications against HTTP Host Header injection attacks by whitelisting valid host domains.
