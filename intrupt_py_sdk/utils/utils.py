from typing import Optional


def _filter_kwargs(kwargs: dict, allowed: Optional[list]) -> dict:
    """Return a copy of `kwargs` containing only keys the approver should see.

    `RunnableConfig` and similar framework plumbing should never reach the
    approver; the tool author opts in to specific keys via `args=[...]`.
    """
    if allowed is None:
        return {k: v for k, v in kwargs.items() if k != "config"}
    return {k: v for k, v in kwargs.items() if k in allowed}