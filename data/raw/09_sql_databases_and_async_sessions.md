# Document ID: fastapi-sql-databases
# Title: Relational SQL Databases and SQLAlchemy Integration
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Persistence & Data Access

## Section 1: SQLAlchemy 2.0 and AsyncEngine
FastAPI integrates seamlessly with SQLAlchemy 2.0 for relational database operations (PostgreSQL, MySQL, SQLite). Using `create_async_engine` and `async_sessionmaker`, database operations run asynchronously using drivers like `asyncpg` for PostgreSQL or `aiosqlite` for SQLite.

## Section 2: Scoped Session Lifecycle via Dependencies
A best practice in FastAPI is providing database sessions via a `get_db` dependency using `yield`. The session is created when the request arrives, injected into the endpoint handler, and closed automatically when the request completes, guaranteeing that uncommitted transactions are rolled back and connections return to the pool.

## Section 3: Alembic Migrations and Schema Versioning
Database schema versioning is managed with Alembic. Alembic inspects SQLAlchemy Declarative models, automatically generates migration scripts comparing target models with current database state, and executes sequential schema upgrades or downgrades across staging and production environments.
