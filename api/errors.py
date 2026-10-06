"""How pipeline failures become HTTP errors.

  422  the input can't be used: an unreadable job page or resume
  409  offline mode and the query isn't cached - nothing was spent
  502  a model or search call failed upstream
"""

from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from sources.exa_client import ExaCacheMiss, PageUnreadable
from sources.resume import ResumeUnreadable

# Not ValueError: pydantic's ValidationError is one, and a model returning
# bad output is an upstream failure, not bad input.
UNUSABLE_INPUT = (PageUnreadable, ResumeUnreadable)


@contextmanager
def upstream(step: str):
    """Wrap a model or search call: input problems keep their own status,
    anything else is the upstream service failing."""
    try:
        yield
    except (HTTPException, ExaCacheMiss, *UNUSABLE_INPUT):
        raise
    except Exception as exc:
        raise HTTPException(502, f"{step} failed: {type(exc).__name__}: {exc}") from exc


def register(app: FastAPI) -> None:
    @app.exception_handler(ExaCacheMiss)
    def cache_miss(_: Request, exc: ExaCacheMiss) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    for error in UNUSABLE_INPUT:
        @app.exception_handler(error)
        def unusable(_: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status_code=422)
