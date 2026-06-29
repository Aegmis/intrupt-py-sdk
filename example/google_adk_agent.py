"""Google ADK example agent with intrupt HITL approval.

Exposes three endpoints:
  POST /call-tool   {"message": str, "session_id": str (optional)}
  POST /resume      {"session_id": str, "approval_id": str, "approved": bool}
  GET  /result/{session_id}

Run:
  source .venv/bin/activate
  APPROVAL_BASE_URL=http://localhost:8080 APPROVAL_API_KEY=sk_org_... \\
  AGENT_PUBLIC_URL=http://localhost:8082 \\
  python example/google_adk_agent.py

Smoke test:
  curl -X POST http://localhost:8082/call-tool \\
    -H "Content-Type: application/json" \\
    -d '{"message": "buy 10 shares of AAPL"}'
"""
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv(override=False)

from google.adk import Agent  # type: ignore[import]
from google.adk.sessions import InMemorySessionService  # type: ignore[import]

from intrupt_py_sdk import ApprovalMiddleware
from intrupt_py_sdk.adapters.google_adk import ApprovalRunner, approval_required

AGENT_PUBLIC_URL = os.environ.get("AGENT_PUBLIC_URL", "http://localhost:8082")
_RESUME_SECRET = os.environ.get("AGENT_RESUME_SECRET", "")

ApprovalMiddleware(
    base_url=os.environ.get("APPROVAL_BASE_URL", "http://localhost:8080"),
    api_key=os.environ.get("APPROVAL_API_KEY", ""),
)

# ── Tools ─────────────────────────────────────────────────────────────────────

def get_stock_price(symbol: str) -> dict:
    """Return a mock stock price for the given ticker symbol."""
    prices = {"AAPL": 189.5, "GOOG": 175.2, "TSLA": 245.0}
    price = prices.get(symbol.upper(), 100.0)
    return {"symbol": symbol.upper(), "price": price, "currency": "USD"}


@approval_required(
    action="purchase_stock",
    message="Human approval required before executing a stock purchase.",
    channel="slack",
    args=["symbol", "quantity"],
)
async def purchase_stock(symbol: str, quantity: int, tool_context=None) -> dict:
    """Execute a stock purchase order for the given symbol and quantity."""
    price = get_stock_price(symbol)["price"]
    total = price * quantity
    return {
        "status": "purchased",
        "symbol": symbol.upper(),
        "quantity": quantity,
        "price_per_share": price,
        "total": total,
    }


# ── Agent & runner ─────────────────────────────────────────────────────────────

_session_service = InMemorySessionService()

_agent = Agent(
    name="finance-agent",
    model="gemini-2.0-flash",
    description="A finance assistant that can look up stock prices and purchase shares.",
    instruction="You are a helpful finance assistant. Use get_stock_price to look up prices before purchasing.",
    tools=[get_stock_price, purchase_stock],
)

runner = ApprovalRunner(
    agent=_agent,
    app_name="finance-bot",
    session_service=_session_service,
    callback_url=f"{AGENT_PUBLIC_URL}/resume",
    callback_secret=_RESUME_SECRET,
)

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Google ADK Example Agent")


class CallToolRequest(BaseModel):
    message: str
    session_id: str = ""


class ResumeRequest(BaseModel):
    session_id: str
    approval_id: str
    approved: bool


@app.post("/call-tool")
async def call_tool(body: CallToolRequest):
    session_id = body.session_id or str(uuid.uuid4())
    return await runner.run(session_id, body.message)


@app.post("/resume")
async def resume(body: ResumeRequest):
    if _RESUME_SECRET:
        from fastapi import Request
        # Secret validation is handled inline in production; skipped for brevity here.
        pass
    return await runner.resume(body.session_id, approved=body.approved, approval_id=body.approval_id)


@app.get("/result/{session_id}")
async def get_result(session_id: str):
    result = runner._results.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No result for this session_id")
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
