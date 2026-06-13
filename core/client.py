import os
import requests
from typing import Optional


class ApprovalClient:

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 10.0):
        """HTTP client for the approval API.

        Args:
            base_url: Base URL of the approval API (defaults to APPROVAL_BASE_URL env var).
            api_key:  Bearer token (defaults to APPROVAL_API_KEY env var).
            timeout:  Per-request HTTP timeout in seconds.
        """
        self.base_url = (base_url or os.environ.get("APPROVAL_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("APPROVAL_API_KEY")
        self.timeout = timeout
        self.hooks: dict = {}

    def add_hook(self, event, fn):
        self.hooks.setdefault(event, []).append(fn)

    def emit(self, event, payload):
        for fn in self.hooks.get(event, []):
            fn(payload)

    def create_approval(
        self,
        *,
        thread_id: str,
        action: str,
        message: str,
        channel: str,
        tool: dict,
        agent_callback_url: Optional[str] = None,
        **metadata,
    ) -> dict:
        """Create a pending approval. Returns {"approval_id": ..., "status": "pending"}.

        `thread_id` is the LangGraph (or other framework) checkpoint id — the API
        stores it so that when the human decides, the approval handler can hit
        the agent's `/resume` with the right context.
        """
        if not thread_id:
            raise ValueError("thread_id is required — needed to resume the paused run")

        response = requests.post(
            f"{self.base_url}/approval",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={
                "thread_id": thread_id,
                "action": action,
                "message": message,
                "channel": channel,
                "tool_name": tool.get("name"),
                "tool_args": list(tool.get("args") or []),
                "tool_kwargs": dict(tool.get("kwargs") or {}),
                "agent_callback_url": agent_callback_url,
                **metadata,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
