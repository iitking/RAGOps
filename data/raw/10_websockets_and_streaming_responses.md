# Document ID: fastapi-websockets-streaming
# Title: WebSockets and Server-Sent Streaming Responses
# Source: FastAPI Documentation (FastAPI by Sebastián Ramírez, Tiangolo, MIT License)
# Category: Realtime Communication

## Section 1: WebSocket Endpoints
FastAPI supports real-time, two-way full-duplex communication using `@app.websocket("/ws")`. The endpoint receives an instance of `fastapi.WebSocket`. The server calls `await websocket.accept()`, continuously listens for messages using `await websocket.receive_text()`, and dispatches replies using `await websocket.send_text()` or `send_json()`.

## Section 2: Streaming Responses with Chunked Transfer
For streaming large payloads such as LLM text generation tokens, video files, or bulk CSV exports, FastAPI provides `StreamingResponse`. It takes an asynchronous or synchronous Python generator yielding chunks of bytes, setting `Transfer-Encoding: chunked` without buffering the complete response in RAM.

## Section 3: Server-Sent Events (SSE)
Server-Sent Events allow servers to push real-time updates unidirectionally over standard HTTP connections using the `text/event-stream` media type. SSE is easier to deploy behind standard reverse proxies than WebSockets and supports automatic client reconnection out of the box.
