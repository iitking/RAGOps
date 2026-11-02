# Document ID: fastapi-query-params
# Title: Query Parameters and String Validations
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Query Handling

## Section 1: Query Parameters and Defaults
Function parameters that are not declared as path parameters are automatically interpreted as URL query parameters. Default values (such as `skip: int = 0` or `limit: int = 10`) make parameters optional. If a parameter has no default value and is not wrapped with `Optional` or `None`, FastAPI treats it as a required query parameter.

## Section 2: Validations with Query()
The `Query` validator provides advanced constraints for query parameters, including `min_length`, `max_length`, `pattern`, `title`, and `description`. For example, `q: Optional[str] = Query(default=None, min_length=3, max_length=50)` validates that if `q` is provided, its length must stay between 3 and 50 characters.

## Section 3: List Query Parameters and Aliases
FastAPI supports receiving multiple values for the same query parameter using `List[str] = Query(default=[])`. If an API requires a parameter name that is invalid syntax in Python (such as `item-query` with a hyphen), the `alias` argument in `Query(alias="item-query")` maps the HTTP query string directly to a valid Python variable name.
