"""network-ai's HTTP API: the existing pipeline behind FastAPI, for the React front end.

No logic lives here - routes load, call the same functions the Streamlit app
calls, save, and return. Request and response bodies are Pydantic models, so
/docs is generated from them and the front end's TypeScript types are
generated from /openapi.json.

Local and single-user, like the rest of the app: no auth, bind to localhost.

Run:  uv run fastapi dev api/main.py        (http://127.0.0.1:8000/docs)
"""

from fastapi import FastAPI
from fastapi.routing import APIRoute

from api import errors, instances, meta, profile


def _operation_id(route: APIRoute) -> str:
    # Function names, not FastAPI's long generated IDs - they become the
    # operation names in the generated TypeScript.
    return route.name


app = FastAPI(title="network-ai", version="0.1.0", generate_unique_id_function=_operation_id)
errors.register(app)
for router in (meta.router, instances.router, profile.router):
    app.include_router(router, prefix="/api")
