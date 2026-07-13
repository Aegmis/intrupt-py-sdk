"""CrewAI example agent with intrupt HITL approval.

Exposes three endpoints:
  POST /call-tool   {"message": str, "run_id": str (optional)}
  POST /resume      {"run_id": str, "approval_id": str, "approved": bool}
  GET  /result/{run_id}

Run:
  source .venv/bin/activate
  OPENAI_API_KEY=sk-... \\
  AEGMIS_BASE_URL=http://localhost:8080 AEGMIS_API_KEY=sk_org_... \\
  AGENT_PUBLIC_URL=http://localhost:8084 \\
  ./run.sh crewai_agent

Smoke test:
  curl -X POST http://localhost:8084/call-tool \\
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

from crewai import Agent, Crew, Task  # type: ignore[import]
from crewai.tools import BaseTool  # type: ignore[import]

from intrupt_py_sdk import ApprovalMiddleware
from intrupt_py_sdk.adapters.crewai import ApprovalCrew, approval_required

AGENT_PUBLIC_URL = os.environ.get("AGENT_PUBLIC_URL", "http://localhost:8084")
_RESUME_SECRET = os.environ.get("AGENT_RESUME_SECRET", "")

ApprovalMiddleware(
    base_url=os.environ.get("AEGMIS_BASE_URL", "http://localhost:8080"),
    api_key=os.environ.get("AEGMIS_API_KEY", ""),
)

# ── Tools ─────────────────────────────────────────────────────────────────────

class StockPriceTool(BaseTool):
    name: str = "get_stock_price"
    description: str = "Return the current price for a ticker symbol."

    def _run(self, symbol: str) -> dict:
        prices = {"AAPL": 189.5, "GOOG": 175.2, "TSLA": 245.0}
        price = prices.get(symbol.upper(), 100.0)
        return {"symbol": symbol.upper(), "price": price, "currency": "USD"}


class PurchaseTool(BaseTool):
    name: str = "purchase_stock"
    description: str = "Execute a stock purchase order for the given symbol and quantity."

    def _run(self, symbol: str, quantity: int) -> dict:
        prices = {"AAPL": 189.5, "GOOG": 175.2, "TSLA": 245.0}
        price = prices.get(symbol.upper(), 100.0)
        return {
            "status": "purchased",
            "symbol": symbol.upper(),
            "quantity": quantity,
            "price_per_share": price,
            "total": price * quantity,
        }

    async def _arun(self, symbol: str, quantity: int) -> dict:
        return self._run(symbol, quantity)


gated_purchase = approval_required(
    PurchaseTool(),
    action="purchase_stock",
    message="Human approval required before executing a stock purchase.",
    channel="slack",
    args=["symbol", "quantity"],
)

# ── Crew setup ─────────────────────────────────────────────────────────────────

_finance_agent = Agent(
    role="Finance Assistant",
    goal="Help users buy stocks after checking prices and obtaining approval.",
    backstory="You are a careful finance assistant who always verifies prices before purchasing.",
    tools=[StockPriceTool(), gated_purchase],
    verbose=False,
)

def _build_crew(request: str) -> Crew:
    task = Task(
        description=f"Handle this finance request: {request}",
        expected_output="A summary of what was done, including purchase confirmation or cancellation.",
        agent=_finance_agent,
    )
    return Crew(agents=[_finance_agent], tasks=[task], verbose=False)


# ── ApprovalCrew wrapper ───────────────────────────────────────────────────────

_approval_crew_cache: dict[str, "ApprovalCrew"] = {}


def _get_approval_crew(request: str, run_id: str) -> "ApprovalCrew":
    crew = _build_crew(request)
    ac = ApprovalCrew(
        crew=crew,
        callback_url=f"{AGENT_PUBLIC_URL}/resume",
        callback_secret=_RESUME_SECRET,
    )
    _approval_crew_cache[run_id] = ac
    return ac


# Shared result/task store so /resume can find the right ApprovalCrew instance.
_results: dict[str, dict] = {}

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="CrewAI Example Agent")


class CallToolRequest(BaseModel):
    message: str
    run_id: str = ""


class ResumeRequest(BaseModel):
    run_id: str
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
    run_id = body.run_id or str(uuid.uuid4())
    ac = _get_approval_crew(body.message, run_id)
    result = await ac.kickoff(run_id, inputs={"request": body.message})
    if result.get("status") in ("complete", "error"):
        _results[run_id] = result
        return _raise_if_error(result)
    # Pending approval — long-poll up to 5 minutes for a human decision via /resume.
    for _ in range(300):
        await asyncio.sleep(1)
        final = ac._results.get(run_id)
        if final and final.get("status") not in ("pending_approval",):
            _results[run_id] = final
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

    ac = _approval_crew_cache.get(body.run_id)
    if ac is None:
        raise HTTPException(status_code=404, detail="Unknown run_id — no pending crew found")
    # resume() is synchronous — returns {"status": "accepted"} immediately so
    # Slack webhooks get a fast 200 OK. The long-poll in /call-tool detects completion.
    return ac.resume(body.run_id, approved=body.approved, approval_id=body.approval_id)


@app.get("/result/{run_id}")
async def get_result(run_id: str):
    result = _results.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No result for this run_id")
    return result


if __name__ == "__main__":
    from urllib.parse import urlparse
    # Listen on the same port advertised to the approval API as the callback host.
    _port = urlparse(AGENT_PUBLIC_URL).port or 8084
    uvicorn.run(app, host="0.0.0.0", port=_port)
