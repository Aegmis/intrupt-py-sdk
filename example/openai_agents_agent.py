"""OpenAI Agents SDK example agent with intrupt HITL approval.

Exposes three endpoints:
  POST /call-tool   {"message": str, "thread_id": str (optional)}
  POST /resume      {"thread_id": str, "approval_id": str, "approved": bool}
  GET  /result/{thread_id}

Run:
  source .venv/bin/activate
  OPENAI_API_KEY=sk-... \\
  AEGMIS_BASE_URL=http://localhost:8080 AEGMIS_API_KEY=sk_org_... \\
  AGENT_PUBLIC_URL=http://localhost:8083 \\
  python example/openai_agents_agent.py

Smoke test:
  curl -X POST http://localhost:8083/call-tool \\
    -H "Content-Type: application/json" \\
    -d '{"message": "buy 10 shares of AAPL"}'
"""
import asyncio
import hmac
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv(override=False)

from agents import Agent, function_tool  # type: ignore[import]

from intrupt_py_sdk import ApprovalMiddleware
from intrupt_py_sdk.adapters.openai_agents import ApprovalAgentRunner, approval_required

AGENT_PUBLIC_URL = os.environ.get("AGENT_PUBLIC_URL", "http://localhost:8083")
_RESUME_SECRET = os.environ.get("AGENT_RESUME_SECRET", "")

ApprovalMiddleware(
    base_url=os.environ.get("AEGMIS_BASE_URL", "http://localhost:8080"),
    api_key=os.environ.get("AEGMIS_API_KEY", ""),
)

# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool
def get_stock_price(symbol: str) -> dict:
    """Return a mock stock price for the given ticker symbol."""
    prices = {"AAPL": 189.5, "GOOG": 175.2, "TSLA": 245.0}
    price = prices.get(symbol.upper(), 100.0)
    return {"symbol": symbol.upper(), "price": price, "currency": "USD"}


@function_tool
@approval_required(
    action="purchase_stock",
    message="Human approval required before executing a stock purchase.",
    channel="slack",
    args=["symbol", "quantity"],
)
async def purchase_stock(symbol: str, quantity: int) -> dict:
    """Execute a stock purchase order for the given symbol and quantity."""
    prices = {"AAPL": 189.5, "GOOG": 175.2, "TSLA": 245.0}
    price = prices.get(symbol.upper(), 100.0)
    return {
        "status": "purchased",
        "symbol": symbol.upper(),
        "quantity": quantity,
        "price_per_share": price,
        "total": price * quantity,
    }


# ── Agent & runner ─────────────────────────────────────────────────────────────

_agent = Agent(
    name="finance-agent",
    instructions=(
        "You are a helpful finance assistant. Use get_stock_price to look up "
        "the current price before purchasing shares."
    ),
    tools=[get_stock_price, purchase_stock],
)

runner = ApprovalAgentRunner(
    agent=_agent,
    callback_url=f"{AGENT_PUBLIC_URL}/resume",
    callback_secret=_RESUME_SECRET,
)

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="OpenAI Agents SDK Example Agent")


class CallToolRequest(BaseModel):
    message: str
    thread_id: str = ""


class ResumeRequest(BaseModel):
    thread_id: str
    approval_id: str
    approved: bool


def _raise_if_error(result: dict) -> dict:
    """Surface an SDK {"status": "error"} result as a real HTTP error.

    The approval API's status (e.g. 422 channel mismatch) rides along in
    "status_code"; without this the endpoint would return HTTP 200 with an
    error body, masking the failure from the caller.
    """
    if result.get("status") == "error":
        raise HTTPException(
            status_code=result.get("status_code", 502),
            detail=result.get("error", "approval failed"),
        )
    return result


@app.post("/call-tool")
async def call_tool(body: CallToolRequest):
    thread_id = body.thread_id or str(uuid.uuid4())
    result = await runner.run(thread_id, body.message)
    if result.get("status") != "pending_approval":
        return _raise_if_error(result)
    # Long-poll up to 5 minutes for the human to approve/reject via /resume.
    for _ in range(300):
        await asyncio.sleep(1)
        final = runner._results.get(thread_id)
        if final and final.get("status") != "pending_approval":
            return _raise_if_error(final)
    raise HTTPException(status_code=408, detail="approval timed out")


@app.post("/resume")
async def resume(body: ResumeRequest, x_agent_secret: str | None = Header(default=None)):
    # Authenticate the approval API's callback when AGENT_RESUME_SECRET is set.
    # Constant-time compare so the secret can't be recovered via response timing.
    if _RESUME_SECRET and not (
        x_agent_secret is not None
        and hmac.compare_digest(x_agent_secret, _RESUME_SECRET)
    ):
        raise HTTPException(status_code=401, detail="missing or invalid X-Agent-Secret")
    return await runner.resume(body.thread_id, approved=body.approved, approval_id=body.approval_id)


@app.get("/result/{thread_id}")
async def get_result(thread_id: str):
    result = runner._results.get(thread_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No result for this thread_id")
    return result


if __name__ == "__main__":
    from urllib.parse import urlparse
    # Listen on the same port advertised to the approval API as the callback host.
    _port = urlparse(AGENT_PUBLIC_URL).port or 8083
    uvicorn.run(app, host="0.0.0.0", port=_port)
