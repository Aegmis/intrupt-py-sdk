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
pip install intrupt-py-sdk
```

Or install with development dependencies:

```bash
pip install intrupt-py-sdk[test]
```

## Quick Start

### 1. Set up the Approval API

First, ensure you have the approval API running. Set the following environment variables:

```bash
export APPROVAL_BASE_URL="http://localhost:8080"
export APPROVAL_API_KEY="your-api-key"
```

### 2. Initialize the Middleware

Initialize the `ApprovalMiddleware` once at application startup:

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

ApprovalMiddleware(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)
```

### 3. Use the Client

Get the client instance and create approvals:

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware

client = ApprovalMiddleware.get_client()

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

See `example/agent.py` for a complete FastAPI + LangGraph agent with approval workflows:

```python
from intrupt_py_sdk.adapters.approval_middleware import ApprovalMiddleware
from intrupt_py_sdk.adapters.langgraph import approval_required

# Initialize middleware
ApprovalMiddleware(base_url=APPROVAL_API_URL, api_key=APPROVAL_API_KEY)

# Define approval-required tool
@tool
@approval_required(
    action="purchase_stock",
    message="Approve buying shares",
    channel="slack",
    args=["symbol", "quantity"],
)
def purchase_stock(symbol: str, quantity: int, config: RunnableConfig) -> dict:
    # Tool implementation
    pass

# Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
agent = graph.compile(checkpointer=memory)
```

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

### ApprovalClient

Direct client usage:

```python
from intrupt_py_sdk.core.client import ApprovalClient

client = ApprovalClient(
    base_url="http://localhost:8080",
    api_key="your-api-key",
    timeout=10.0
)
```


### approval_required Decorator

```python
@approval_required(
    action="action_name",           # Optional, defaults to function name
    message="Approval required",    # Optional, defaults to generic message
    channel="slack",                # Optional, defaults to "slack"
    args=["arg1", "arg2"],         # Optional, kwargs to include in approval
)
def my_tool(arg1: str, arg2: int, config: RunnableConfig) -> dict:
    pass
```


## Development

### Running Tests

```bash
pytest
```

### Running the Example Agent

```bash
cd example
python agent.py
```

The example agent runs on `http://localhost:8081` and provides:
- `POST /call-tool`: Start or continue a chat
- `POST /resume`: Resume an approval-paused run

## License

Apache 2.0
