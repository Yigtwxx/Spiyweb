"""Finding the built browser bundle, and saying so plainly when it is absent.

The front end is compiled TypeScript, so it cannot live in the source tree
the way Python does: `web/dist` is built by the front-end toolchain and
force-included into the wheel as `spiyweb/viewer/static`. That leaves three
situations, and the whole job of this module is to tell them apart instead of
failing with a stack trace about a missing directory:

- **Installed wheel** - the bundle is package data and this just works.
- **Repository checkout** - nothing is packaged yet; the caller points at
  their own `web/dist` (or sets `SPIYWEB_VIEWER_BUNDLE`). Deliberately NOT a
  silent walk up the tree looking for a repository: the package guessing that
  it might be inside a checkout is the habit Faz 2.2 deleted.
- **Neither** - the API still serves JSON; only the page is missing, and the
  error says which of the two situations the caller is in.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["BUNDLE_ENV", "MissingBundle", "bundle_path", "find_bundle"]

BUNDLE_ENV = "SPIYWEB_VIEWER_BUNDLE"
"""Points at a `web/dist` for people working from a checkout."""

STATIC_DIRNAME = "static"
"""Where the wheel keeps the bundle, relative to this package."""

ENTRY = "index.html"
"""The file whose presence means "this really is a built bundle"."""


class MissingBundle(RuntimeError):
    """No browser bundle to serve; the message says what to do about it."""


def bundle_path() -> Path | None:
    """The packaged bundle, the environment override, or `None`.

    Order matters: the override wins, so a developer running from a checkout
    against an installed spiyweb sees their own build rather than a stale
    packaged one.
    """
    override = os.environ.get(BUNDLE_ENV, "").strip()
    if override:
        candidate = Path(override)
        return candidate if (candidate / ENTRY).is_file() else None
    packaged = Path(__file__).resolve().parent / STATIC_DIRNAME
    return packaged if (packaged / ENTRY).is_file() else None


def find_bundle(explicit: Path | str | None = None) -> Path:
    """The bundle to serve, or `MissingBundle` explaining which case this is."""
    if explicit is not None:
        candidate = Path(explicit)
        if not (candidate / ENTRY).is_file():
            raise MissingBundle(
                f"{candidate} does not look like a built browser bundle "
                f"(no {ENTRY} in it); run `npm run build` in web/ first"
            )
        return candidate
    found = bundle_path()
    if found is None:
        raise MissingBundle(
            "this spiyweb has no browser bundle packaged with it. From a "
            "repository checkout, build it once with `npm ci && npm run "
            f"build` in web/ and point {BUNDLE_ENV} at web/dist (or pass "
            "bundle=...). From an installed wheel, this means the wheel was "
            "built without the front end - the API still works, only the "
            "page is missing"
        )
    return found
