"""Repository paths, and the one `sys.path` adjustment this server makes.

`ui/graph_view.py` is a pure module (numpy only, no Streamlit, no I/O) that
already carries the layout, scene and comparison logic, with 500+ tests
pinning it. The server reuses it rather than reimplementing any of it in
TypeScript, so one query produces one picture no matter which front end asked.

`ui/` is not an importable package - deliberately, so `pip install spiyweb`
never drags Streamlit in - which is why `pyproject.toml` already tells pytest
`pythonpath = ["ui"]` with the note "one canonical module name, both contexts".
This module is the third context, applying the same rule in one visible place
instead of scattering path hacks.

Promotion trigger, written down so it is a decision and not a drift: if this
server ever needs to be installable outside the repository, or a fourth
consumer appears, move `ui/graph_view.py` to `src/spiyweb/scene.py` and open a
`spiyweb[view] = ["numpy"]` extra. Until then, a repo-rooted localhost tool
does not justify the churn of breaking every `import graph_view` in the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / "ui"
DATA_ROOT = REPO_ROOT / "data"
WEB_DIST = REPO_ROOT / "web" / "dist"
RUN_LOCK = DATA_ROOT / ".run.lock"
LOG_DIR = DATA_ROOT / ".runs"


def ensure_ui_importable() -> None:
    """Put `ui/` on `sys.path` once, so `import graph_view` works here too."""
    entry = str(UI_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)


ensure_ui_importable()
