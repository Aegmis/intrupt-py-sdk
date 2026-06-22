import os

# pyproject.toml's `pythonpath = ["..", "../example"]` puts the repo root
# and example/ on sys.path so `intrupt_py_sdk` (namespace package) and the
# example `agent` module are both importable from inside these tests.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("APPROVAL_BASE_URL", "http://localhost-test")
os.environ.setdefault("APPROVAL_API_KEY", "sk_org_org_test1234_abcdef0123456789")
