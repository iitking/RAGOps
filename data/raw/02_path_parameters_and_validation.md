# Document ID: fastapi-path-params
# Title: Path Parameters, Type Hints and Validations
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Routing and Validation

## Section 1: Declaring Path Parameters
Path parameters are declared using Python format string syntax inside the path decorator, such as `@app.get("/items/{item_id}")`. The function parameter name must match the path variable. Python type hints (such as `item_id: int`) enable automatic string conversion, type validation, HTTP 422 error generation for invalid inputs, and OpenAPI parameter documentation.

## Section 2: Path Validation with Path()
FastAPI provides the `Path` function from `fastapi` to declare metadata and numerical validations on path parameters. Developers can enforce integer boundaries using `ge` (greater than or equal to), `gt` (greater than), `le` (less than or equal to), and `lt` (less than). String path parameters can also enforce regular expression patterns using the `regex` or `pattern` parameter.

## Section 3: Predefined Path Values Using Enum
When path parameters should only accept a fixed set of valid choices, Python's standard `Enum` class can be used. By inheriting from both `str` and `Enum`, FastAPI serializes enum members as strings in OpenAPI specifications, providing an interactive dropdown in Swagger UI documentation.
