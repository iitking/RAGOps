# Document ID: fastapi-testing
# Title: Automated API Testing with Pytest and HTTPX
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Quality Assurance & Testing

## Section 1: TestClient vs AsyncClient
FastAPI provides `TestClient` (built on HTTPX) for synchronous end-to-end testing of path operations without spinning up a live network port. For testing asynchronous database sessions or WebSocket endpoints, `httpx.AsyncClient` with `ASGITransport(app=app)` allows running native `async def test_*` test functions under `pytest-asyncio`.

## Section 2: Overriding Dependencies for Testing
FastAPI allows replacing production dependencies (such as external authentication services or production databases) using `app.dependency_overrides[get_db] = override_get_db`. This isolates tests against an in-memory SQLite database or mock authentication token without modifying application source code.

## Section 3: Test Fixtures and Database Rollbacks
Pytest fixtures with `yield` set up fresh database schemas before each test run and roll back transactions after each test concludes. This guarantees zero state leakage across test suites, ensuring reproducible and deterministic continuous integration builds.
