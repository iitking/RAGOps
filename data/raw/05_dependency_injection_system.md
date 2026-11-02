# Document ID: fastapi-dependencies
# Title: FastAPI Dependency Injection System
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Architecture & Modularity

## Section 1: Overview of Depends
FastAPI features a hierarchical Dependency Injection system built around `fastapi.Depends`. Dependencies allow developers to share database sessions, enforce authentication, perform query sanitization, and share business logic across multiple path operations without boilerplate repetition.

## Section 2: Sub-dependencies and Resolution Graphs
Dependencies can themselves declare other dependencies, forming a directed acyclic graph (DAG) of requirements. FastAPI resolves all sub-dependencies in topological order, passes resolved values to downstream dependents, and caches results within a single request scope (unless `use_cache=False` is explicitly set).

## Section 3: Dependencies with yield (Resource Teardown)
FastAPI supports context-manager style dependencies using the `yield` keyword. Code before the `yield` statement executes prior to running the path operation (e.g., establishing a database connection), and code after `yield` executes after the response is returned to the client (e.g., closing transactions and releasing connections back to the pool).
