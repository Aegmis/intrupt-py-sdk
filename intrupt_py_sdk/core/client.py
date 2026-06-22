import os
import requests
from typing import Optional


class ApprovalClient:

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 10.0):
        """HTTP client for the approval API.

        Args:
            base_url: Base URL of the approval API (defaults to APPROVAL_BASE_URL env var).
            api_key:  API key for authentication (format: sk_org_{org_id}_{hash}).
                     Defaults to APPROVAL_API_KEY env var.
            timeout:  Per-request HTTP timeout in seconds.
        """
        self.base_url = (base_url or os.environ.get("APPROVAL_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("APPROVAL_API_KEY")
        self.timeout = timeout
        self.hooks: dict = {}
        self._org_id = self._extract_org_id_from_api_key()

    def _extract_org_id_from_api_key(self) -> Optional[str]:
        """Extract org_id from API key format: sk_org_{org_id}_{hash}

        Example: sk_org_org_0819dfb9_<hash> → org_0819dfb9
        """
        if not self.api_key:
            raise ValueError("API key is required but not provided")

        # Expected format: sk_org_{org_id}_{hash}
        # The hash is always the last 16 hex characters
        if not self.api_key.startswith("sk_org_"):
            raise ValueError(
                f"Invalid API key format. Expected 'sk_org_{{org_id}}_{{hash}}', got '{self.api_key[:20]}...'"
            )

        # Remove "sk_org_" prefix and find the last underscore (separator before hash)
        after_prefix = self.api_key[7:]  # Remove "sk_org_"

        # The hash is the last 16 characters (16 hex chars from uuid.hex[:16])
        # Find the last underscore - everything before it is org_id
        last_underscore_idx = after_prefix.rfind("_")

        if last_underscore_idx == -1:
            raise ValueError(
                f"Invalid API key format. Expected 'sk_org_{{org_id}}_{{hash}}', got '{self.api_key[:20]}...'"
            )

        org_id = after_prefix[:last_underscore_idx]

        if not org_id or not org_id.startswith("org_"):
            raise ValueError(
                f"Invalid org_id in API key. Expected 'org_*', got '{org_id}'"
            )

        return org_id

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
        agent_callback_secret: Optional[str] = None,
        **metadata,
    ) -> dict:
        """Create a pending approval. Returns {"approval_id": ..., "status": "pending"}.

        `thread_id` is the LangGraph (or other framework) checkpoint id — the API
        stores it so that when the human decides, the approval handler can hit
        the agent's `/resume` with the right context.

        Organization ID is automatically extracted from the API key.
        """
        if not thread_id:
            raise ValueError("thread_id is required — needed to resume the paused run")

        # Always use org-scoped endpoint. org_id is extracted from API key
        endpoint = f"{self.base_url}/org/{self._org_id}/approval"

        response = requests.post(
            endpoint,
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
                "agent_callback_secret": agent_callback_secret,
                **metadata,
            },
            timeout=self.timeout,
        )
        if not response.ok:
            try:
                detail = response.json().get("detail") or response.text
            except Exception:
                detail = response.text
            raise requests.exceptions.HTTPError(
                f"{response.status_code} Error: {detail}",
                response=response,
            )
        return response.json()
