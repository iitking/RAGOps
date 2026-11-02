# Document ID: fastapi-security
# Title: Security, OAuth2, and JWT Bearer Authentication
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Security & Authentication

## Section 1: OAuth2 with Password Flow
FastAPI provides `OAuth2PasswordBearer` and `OAuth2PasswordRequestForm` to implement the OAuth2 password grant flow with Bearer tokens. Clients send credentials (`username` and `password`) to a `/token` endpoint using form data, and the server returns a signed access token.

## Section 2: JSON Web Tokens (JWT) Generation and Validation
JWT tokens are compact, cryptographically signed claims serialized as base64-encoded JSON. The server signs tokens using symmetric algorithms like HMAC-SHA256 (HS256) with a secret key or asymmetric algorithms like RSA (RS256). The token payload typically includes `sub` (subject/user identifier), `exp` (expiration timestamp), and optional permission scopes.

## Section 3: Protecting Endpoints with Security Dependencies
Path operations enforce authentication by declaring a dependency on a `get_current_user` function. If the token is missing, expired, or possesses an invalid signature, the dependency raises an `HTTPException` with status code `401 Unauthorized` and header `WWW-Authenticate: Bearer`.
