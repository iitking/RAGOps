# Document ID: fastapi-request-body
# Title: Request Body and Pydantic Schemas
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Data Serialization

## Section 1: Declaring Request Bodies
To declare a request body, developers create a subclass of `pydantic.BaseModel`. FastAPI automatically parses JSON payloads from HTTP requests, converts types to Python primitives, validates field types against model definitions, and raises an HTTP 422 Unprocessable Entity error with detailed field error locations if payload validation fails.

## Section 2: Nested Models and Deep Structures
Pydantic models can be nested hierarchically to represent complex JSON objects. A model can contain attributes that are other Pydantic models, lists of models, or dictionaries. For instance, an `Order` model can contain a `List[OrderItem]` and a `ShippingAddress` model, maintaining full validation across all sub-fields and generating nested schemas in Swagger documentation.

## Section 3: Extra Field Validation and Field Constraints
Field-level validations are declared using `pydantic.Field`. Developers can specify custom error messages, numerical ranges with `gt` and `le`, regex formatting, and example payloads for OpenAPI documentation. Setting `extra = "forbid"` in the model configuration causes FastAPI to reject any unexpected or unannounced fields submitted by the client.
