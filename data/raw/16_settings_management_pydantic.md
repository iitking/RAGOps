# Document ID: fastapi-settings
# Title: Application Settings and Configuration with Pydantic
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Configuration Architecture

## Section 1: BaseSettings for Environment Variables
Configuration settings in modern FastAPI applications are managed using `pydantic_settings.BaseSettings`. Attributes defined on the settings class automatically read corresponding environment variables, perform type coercion (converting strings to integers, booleans, or lists), and validate mandatory configuration secrets on startup.

## Section 2: .env File Support and Loading Priority
`BaseSettings` supports loading values from `.env` files via `model_config = SettingsConfigDict(env_file=".env")`. The loading priority strictly adheres to: system environment variables first, `.env` file variables second, and default class values last.

## Section 3: Dependency Caching with @lru_cache
To prevent re-reading disk environment variables on every incoming HTTP request, developers wrap the `get_settings()` factory function with `@functools.lru_cache()`. When passed to `Depends(get_settings)`, FastAPI reuses the cached settings instance across all requests throughout the server lifespan.
