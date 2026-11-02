# Document ID: fastapi-openapi-schema
# Title: OpenAPI Standards, JSON Schema, and Documentation
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: API Specifications

## Section 1: Automatic OpenAPI Generation
FastAPI automatically parses all registered routes, path parameters, query parameters, headers, cookies, request bodies, and response models to produce a strictly compliant OpenAPI 3.1.0 JSON schema accessible at `/openapi.json`.

## Section 2: Interactive Documentation UIs
By default, FastAPI exposes two interactive documentation interfaces generated from the OpenAPI schema: Swagger UI at `/docs` (providing an interactive "Try it out" HTTP client) and ReDoc at `/redoc` (providing a clean, publication-ready reference manual).

## Section 3: Customizing API Metadata and Security Schemes
API developers can customize root documentation metadata including `title`, `description`, `version`, `terms_of_service`, `contact`, and `license`. Custom OpenAPI tags group related endpoints into organized visual sections in Swagger UI.
