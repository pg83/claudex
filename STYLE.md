# Code Style

## Import ordering

Imports are split into three sections, separated by a single blank line.

```python
import os          # 1. Plain `import X` first, sorted by line length
import sys
import json
import time
import uuid
import httpx
import uvicorn
import argparse

from contextlib import asynccontextmanager    # 2. `from X import Y` — one block
                                              # per top-level module
from typing import AsyncIterator, Optional, Union

from starlette.routing import Route           # imports sharing a parent package
from starlette.requests import Request        # are grouped together without a
from starlette.applications import Starlette  # blank line, sorted by module
from starlette.responses import JSONResponse, StreamingResponse  # name length

import claudex.log as lg                      # 3. Local package imports last,
import claudex.common as cx                   # always via `import X as alias`
import claudex.rag as rag_mod                 # with short aliases
```

Rules:
- `import X` statements are sorted by line length.
- Standalone `from X import Y` lines from unrelated modules are separated by
  blank lines.
- `from X import Y` lines sharing a parent package (e.g. `starlette.*`) form
  one contiguous block, sorted by module name length.
- Local modules are imported as `import claudex.X as alias` with a short alias.

## Blank lines between blocks

**Top-level definitions** (functions, classes) are separated by two blank
lines:

```python
def foo():
    ...


def bar():
    ...
```

**Inside functions**, compound statements (`if` / `elif` / `else` / `while` /
`for` / `try` / `except` / `with`) are surrounded by a single blank line —
**but only when the block is not at the very start or end of the enclosing
block**:

```python
def extract_system_text(system) -> str:
    if isinstance(system, str):           # first statement in the function —
        return system                     # no blank line before

    if isinstance(system, list):          # in the middle — blank lines on
        return "\n\n".join(...)           # both sides

    return str(system) if system else ""  # last statement — no blank line after
```

## Blank lines inside blocks

Separate an assignment from a subsequent block with a blank line:

```python
listen = cfg.get("listen", "127.0.0.1:8082")

if ":" in listen:
    ...
```

Separate logical sections inside a function (setup / main work / teardown)
with blank lines:

```python
@asynccontextmanager
async def lifespan(app: Starlette):
    global http_client

    http_client = httpx.AsyncClient(...)

    yield

    await http_client.aclose()
```

Put a blank line before a trailing `return` that follows a block or a
non-trivial expression:

```python
def sse_event(event_type: str, data: dict) -> str:
    data.setdefault("type", event_type)

    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
```
