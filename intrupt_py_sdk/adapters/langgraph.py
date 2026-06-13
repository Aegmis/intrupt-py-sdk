from functools import wraps
from langgraph.types import interrupt
from ..utils import _filter_kwargs



def approval_required(**configs):
    """Decorate a tool so it pauses for human approval before executing.

    The decorated function pauses via `langgraph.types.interrupt(...)`, which
    commits the checkpoint *before* any external side-effect (Slack DM, email)
    is fired. The agent's `/call-tool` handler observes the interrupt, creates
    an approval record on the API, and the human's decision flows back through
    `/resume`. The tool body only runs if the decision is `approved`.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            payload = {
                "approval_required": True,
                "action": configs.get("action", func.__name__),
                "message": configs.get(
                    "message", f"Approval required for {func.__name__}"
                ),
                "channel": configs.get("channel", "slack"),
                "tool": {
                    "name": func.__name__,
                    "description": func.__doc__,
                    "kwargs": _filter_kwargs(kwargs, configs.get("args")),
                },
            }

            decision = interrupt(payload)

            if not isinstance(decision, dict) or not decision.get("approved"):
                return {
                    "status": "cancelled",
                    "tool": func.__name__,
                    "message": f"{func.__name__} was not approved",
                }

            return func(*args, **kwargs)

        return wrapper

    return decorator


