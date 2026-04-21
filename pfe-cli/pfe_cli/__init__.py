"""PFE CLI package."""

__all__ = ["app", "main"]


def __getattr__(name: str):
    if name in {"app", "main"}:
        from .main import app, main

        return {"app": app, "main": main}[name]
    raise AttributeError(name)
