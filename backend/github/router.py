"""GitHub App install flow.

GitHub does not sign its install callback. Anyone can navigate to
``/github/callback?installation_id=12345`` with a guessed number, so the only
integrity check available on that request is one we issue ourselves: the
single-use, short-lived `state` handed out by ``/github/install``. The
installation lookup against the API then confirms the id is really ours.
"""
import secrets
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from .auth import AppCredentials, GitHubConfigError, InstallationNotFound, fetch_installation

router = APIRouter(prefix="/github", tags=["github"])

_STATE_TTL = 600.0

_pending_states: Dict[str, float] = {}
_installations: Dict[int, Dict[str, Any]] = {}
# ponytail: in-process dicts, lost on restart (recovery is to visit
# /github/install again). redis is already a declared dependency and unused —
# that is the upgrade path if restart-survival ever matters.


def issue_state(now: Optional[float] = None) -> str:
    """Mint a single-use state token and record its deadline."""
    now = time.monotonic() if now is None else now
    state = secrets.token_urlsafe(32)
    _pending_states[state] = now + _STATE_TTL
    # Sweep expired entries so the dict cannot grow without bound.
    for stale, deadline in list(_pending_states.items()):
        if deadline <= now:
            del _pending_states[stale]
    return state


def consume_state(state: str, now: Optional[float] = None) -> bool:
    """Take a state token, returning whether it was valid and unused."""
    now = time.monotonic() if now is None else now
    deadline = _pending_states.pop(state, None)
    return deadline is not None and deadline > now


@router.get("/install")
async def install() -> RedirectResponse:
    """Send the browser to GitHub's install page for this App."""
    try:
        credentials = AppCredentials()
        url = credentials.install_url(issue_state())
    except GitHubConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RedirectResponse(url, status_code=302)


@router.get("/callback")
async def callback(
    installation_id: Optional[str] = Query(default=None),
    setup_action: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Receive GitHub's post-install redirect and verify the installation.

    ``setup_action`` is ``install`` or ``update`` (a re-installation after a
    permissions change). Both are handled identically; the value is echoed back
    so the caller can tell them apart.
    """
    if not state or not consume_state(state):
        raise HTTPException(status_code=400, detail="invalid or expired state")
    if not installation_id or not installation_id.isdigit():
        raise HTTPException(
            status_code=400, detail="missing or malformed installation_id"
        )

    parsed_id = int(installation_id)
    try:
        info = await fetch_installation(AppCredentials(), parsed_id)
    except InstallationNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GitHubConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    account = info.get("account") or {}
    record = {
        "installation_id": parsed_id,
        "account": account.get("login"),
        "account_type": account.get("type"),
        "status": "installed",
        "setup_action": setup_action,
    }
    _installations[parsed_id] = record
    return record


@router.get("/installations")
async def list_installations() -> Dict[str, Any]:
    """The installations seen this process's lifetime."""
    return {"installations": list(_installations.values())}
