"""Repository paths this server reads.

There is no `sys.path` adjustment here any more. `ui/graph_view.py` used to be
a repo-rooted module that both front ends imported by its bare name, and this
file carried the one path hack that made that work outside Streamlit. Its own
docstring named the condition for ending that: *"if this server ever needs to
be installable outside the repository, or a fourth consumer appears, move
`ui/graph_view.py` to `src/spiyweb/scene.py`"*. Both happened, so the module
was promoted and the hack deleted rather than left to rot into a habit.

What remains is repository geography: where the data lives, where the built
browser bundle lives, where a run writes its lock and logs. All of it is
repo-rooted on purpose - this package is the harness-facing server, not the
one that ships inside the wheel.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
WEB_DIST = REPO_ROOT / "web" / "dist"
RUN_LOCK = DATA_ROOT / ".run.lock"
LOG_DIR = DATA_ROOT / ".runs"
