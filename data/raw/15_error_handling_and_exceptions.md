# Document ID: fastapi-error-handling
# Title: Error Handling, HTTPExceptions, and Custom Exception Handlers
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Error Handling & Resilience

## Section 1: Raising HTTPException
To return standard HTTP error responses to clients, FastAPI provides `fastapi.HTTPException`. It accepts a `status_code` (such as 404 Not Found or 400 Bad Request), a `detail` string or dictionary describing the issue, and optional HTTP `headers`.

## Section 2: Custom Exception Handlers
Applications can define custom business exceptions (e.g., `ItemNotFoundError` or `InsufficientBalanceError`) and register global handlers using `@app.exception_handler(CustomException)`. The handler intercepts the error and returns a customized JSON response with structured error payloads.

## Section 3: Overriding Request Validation Errors
When client requests violate Pydantic schema validation rules, FastAPI internally raises `RequestValidationError`. By registering an exception handler for `RequestValidationError`, developers can customize the default 422 JSON response into an enterprise error schema format required by frontend applications.
