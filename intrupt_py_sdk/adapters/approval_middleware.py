import os
from typing import Optional

from intrupt_py_sdk.core.client import ApprovalClient


class ApprovalMiddleware:
    """Holds a process-wide ApprovalClient so tools can reach the API without
    threading the config through every call site.

    Singleton on first construction; subsequent constructions are no-ops so
    that re-importing the module (or instantiating from multiple call sites)
    does not silently re-point the shared client.
    """

    _instance: Optional["ApprovalMiddleware"] = None

    def __new__(cls, base_url: Optional[str] = None, api_key: Optional[str] = None):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst.client = ApprovalClient(
                base_url=base_url or os.environ.get("APPROVAL_BASE_URL"),
                api_key=api_key or os.environ.get("APPROVAL_API_KEY"),
            )
            cls._instance = inst
        return cls._instance

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # Idempotent: see __new__. Re-instantiation must not mutate the shared client.
        pass

    @classmethod
    def get_client(cls) -> ApprovalClient:
        if cls._instance is None:
            raise RuntimeError(
                "ApprovalMiddleware not initialised — construct it once at startup "
                "with the base_url and api_key for the approval API."
            )
        return cls._instance.client
