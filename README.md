# intrupt-py-sdk

Client SDK + framework adapters for the human-in-the-loop approval API.

## Overview

`intrupt-py-sdk` provides a Python client and framework adapters for integrating human approval workflows into AI agents. It includes:

- **HTTP Client**: Direct API client for creating and managing approvals
- **ApprovalMiddleware**: Process-wide singleton for easy client access
- **LangGraph Adapter**: Decorator-based approval integration for LangGraph agents
- **Policy Engine**: Configurable approval policies based on tool names and risk levels

## Installation

```bash
uv add intrupt-py-sdk or pip install intrupt-py-sdk
```

Or install with development dependencies:

```bash
uv add intrupt-py-sdk[test] or pip install intrupt-py-sdk[test]
```

## Quick Start

### 1. Set up the Approval API

First, ensure you have the approval API running. Set the following environment variables:

```bash
export APPROVAL_BASE_URL="http://localhost:8080"    # intrupt API base URL
export APPROVAL_API_KEY="your-api-key" # Optional for self-hosted intrupt API

export AGENT_RESUME_SECRET="your-secret" # Secret for agent resume endpoint, it securely authenticates the resume callback
```

### 2. Initialize the Middleware

Initialize the `ApprovalMiddleware` once at application startup:

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

ApprovalMiddleware(
    base_url="http://localhost:8080", # intrupt API base URL
    api_key="your-api-key" # Optional for self-hosted intrupt API
)
```

### 3. Use the Client

Get the client instance and create approvals:

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.core.client import ApprovalClient

# Get the singleton client instance
client = ApprovalMiddleware.get_client()

# Or create a new client instance
client = ApprovalClient(
    base_url="http://localhost:8080",
    api_key="your-api-key",
    timeout=10.0
)

approval = client.create_approval(
    thread_id="thread-123",
    action="purchase_stock",
    message="Approve buying shares",
    channel="slack",
    tool={
        "name": "purchase_stock",
        "kwargs": {"symbol": "AAPL", "quantity": 10}
    },
    agent_callback_url="http://localhost:8081/resume"
)
```

## LangGraph Integration

The SDK provides a decorator for LangGraph tools that require human approval:

```python
from langchain_core.tools import tool
from langgraph.types import RunnableConfig
from intrupt_py_sdk.adapters.langgraph import approval_required

@tool
@approval_required(
    action="purchase_stock",
    message="Approve buying shares",
    channel="slack",
    args=["symbol", "quantity"],
)
def purchase_stock(symbol: str, quantity: int, config: RunnableConfig) -> dict:
    """Simulate purchasing a given quantity of a stock symbol."""
    return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
    }
```

When the decorated tool is called:
1. The agent pauses via `langgraph.types.interrupt()`
2. An approval record is created on the API
3. The tool body only executes if the human approves

### Complete LangGraph Example

See `example/agent.py` for a complete FastAPI + LangGraph agent with approval workflows

## Configuration

### Environment Variables

- `APPROVAL_BASE_URL`: Base URL of the approval API (default: empty string)
- `APPROVAL_API_KEY`: Bearer token for API authentication (default: empty string)

### ApprovalMiddleware

The `ApprovalMiddleware` is a singleton that holds a process-wide `ApprovalClient`. Initialize it once at startup:

```python
ApprovalMiddleware(
    base_url="http://localhost:8080",  # Optional, defaults to APPROVAL_BASE_URL
    api_key="your-api-key"              # Optional, defaults to APPROVAL_API_KEY
)
```


## Development

```
uv sync
```
### Running Tests

```bash
uv run pytest -v
```

### Running the Example Agent

```bash
uv run python example/agent.py
```

The example agent runs on `http://localhost:8081` and provides:
- `POST /call-tool`: Start or continue a chat
- `POST /resume`: Resume an approval-paused run

