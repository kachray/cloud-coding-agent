"""GitHub App authentication: JWT minting and installation-token exchange.

The credential shape is fixed by CLAUDE.md's security rule — GitHub App
installation tokens only, never a stored PAT or refresh token. Nothing here
persists a credential: the App private key is read from disk, mints a JWT that
lives for minutes, and that JWT buys an installation token that lives for an
hour and is held in memory only.

The user-authorization (OAuth) leg of the App flow is deliberately absent. It
exists to read user metadata (name, email); repo access alone does not need it,
and taking a long-lived refresh token without that need is exactly what
CLAUDE.md rules out.
"""
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx2
import jwt

BACKEND_ROOT = Path(__file__).resolve().parent.parent

_API = "https://api.github.com"
_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# GitHub rejects a JWT whose `exp` is more than 600s out, and one whose `iat`
# is in its future. Backdating plus a sub-limit lifetime means ordinary clock
# skew between here and GitHub cannot push a JWT past either bound.
_JWT_BACKDATE = 60
_JWT_LIFETIME = 540

# A token is never handed out with less than this much life left, so an
# operation that starts seconds before the hour mark still has a valid token
# for its whole duration.
_REFRESH_SKEW = 300

_REQUEST_TIMEOUT = 30.0


class GitHubConfigError(RuntimeError):
    """Required GitHub App configuration is missing or invalid."""


class InstallationNotFound(GitHubConfigError):
    """The installation id does not belong to this App."""


def _parse_expiry(value: str) -> float:
    """GitHub returns UTC ISO-8601, e.g. ``2026-09-21T12:00:00Z``."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    ).timestamp()


class AppCredentials:
    """App-level identity: the id, the private key, and the install URL slug.

    Env is read at call time, not import time — the same lazy pattern
    ``AgentLoop.client`` uses for ``GROQ_API_KEY`` — so it does not matter
    whether ``load_dotenv`` has run before this module is imported.

    The private key comes from a *file path*, never an inline PEM. The sandbox
    spawns shell subprocesses with no ``env=``, so anything in the backend's
    environment is readable from inside the agent's own shell; a private key
    there would be one `cat` away from the model's context.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> None:
        self._app_id = app_id
        self._private_key_path = private_key_path
        self._slug = slug
        self._private_key: Optional[str] = None

    @property
    def app_id(self) -> str:
        # GitHub requires `iss` to be a string even though the id is numeric.
        value = self._app_id or os.environ.get("GITHUB_APP_ID")
        if not value:
            raise GitHubConfigError("GITHUB_APP_ID is not set")
        return str(value)

    @property
    def slug(self) -> str:
        value = self._slug or os.environ.get("GITHUB_APP_SLUG")
        if not value:
            raise GitHubConfigError("GITHUB_APP_SLUG is not set")
        return value

    @property
    def private_key(self) -> str:
        if self._private_key is None:
            configured = self._private_key_path or os.environ.get(
                "GITHUB_APP_PRIVATE_KEY_PATH"
            )
            if not configured:
                raise GitHubConfigError("GITHUB_APP_PRIVATE_KEY_PATH is not set")
            path = Path(configured)
            if not path.is_absolute():
                path = BACKEND_ROOT / path
            if not path.is_file():
                raise GitHubConfigError(
                    f"GitHub App private key not found at {path}"
                )
            self._private_key = path.read_text(encoding="utf-8")
        return self._private_key

    def install_url(self, state: str) -> str:
        """Where to send a browser to install the App, carrying the state."""
        return (
            f"https://github.com/apps/{self.slug}"
            f"/installations/new?state={state}"
        )

    def mint_jwt(self) -> str:
        """A short-lived RS256 JWT proving we are the App itself."""
        now = int(time.time())
        claims = {
            "iss": self.app_id,
            "iat": now - _JWT_BACKDATE,
            "exp": now + _JWT_LIFETIME,
        }
        return jwt.encode(claims, self.private_key, algorithm="RS256")


async def fetch_installation(
    credentials: AppCredentials,
    installation_id: int,
    client: Optional[httpx2.AsyncClient] = None,
) -> Dict[str, Any]:
    """Confirm *installation_id* is a real installation of *our* App.

    This is the check that matters in the callback. A `state` we issued proves
    the request came from a flow we started; it does not prove the
    `installation_id` on that request is ours. An installation belonging to a
    different App 404s here.
    """
    owns_client = client is None
    client = client or httpx2.AsyncClient(timeout=_REQUEST_TIMEOUT)
    try:
        response = await client.get(
            f"{_API}/app/installations/{int(installation_id)}",
            headers={
                **_API_HEADERS,
                "Authorization": f"Bearer {credentials.mint_jwt()}",
            },
        )
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code == 404:
        raise InstallationNotFound(
            f"installation {installation_id} does not belong to this App"
        )
    if response.status_code != 200:
        # Never echo the response body: it is not needed to diagnose this, and
        # error bodies are the one place a credential could plausibly appear.
        raise GitHubConfigError(
            f"GitHub rejected the installation lookup: HTTP {response.status_code}"
        )
    return response.json()


class InstallationTokens:
    """Mints and caches installation tokens, refreshing at the point of use.

    There is no timer and no background task. Every operation that needs a
    credential calls ``get`` immediately before it runs, so a session running
    past the hour mark simply mints a new token on its next call. A background
    refresher would be a second source of truth that can stop silently, and the
    first symptom would be a push failing an hour later for no visible reason —
    lazy refresh cannot be stale at the point of use, because the point of use
    is the only place it exists.
    """

    def __init__(
        self,
        credentials: Optional[AppCredentials] = None,
        client: Optional[httpx2.AsyncClient] = None,
        now=time.time,
    ) -> None:
        self._credentials = credentials or AppCredentials()
        self._client = client
        self._now = now
        self._cache: Dict[int, Tuple[str, float]] = {}
        # ponytail: one global lock; per-installation locks if concurrent
        # sessions ever matter. A duplicate mint is harmless anyway — it does
        # not invalidate the previous token, and GitHub allows 5,000/hour.
        self._lock = asyncio.Lock()

    @property
    def client(self) -> httpx2.AsyncClient:
        if self._client is None:
            self._client = httpx2.AsyncClient(timeout=_REQUEST_TIMEOUT)
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client and forget it, so the property can reopen.

        Clearing the field is what makes closing safe: without it, any later
        ``get()`` would reuse a closed client and raise.
        """
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def get(self, installation_id: int) -> str:
        """A currently-valid installation token, minting one if needed."""
        cached = self._cache.get(installation_id)
        if cached is not None and self._now() < cached[1] - _REFRESH_SKEW:
            return cached[0]
        async with self._lock:
            # Re-check under the lock: a concurrent caller may have just minted.
            cached = self._cache.get(installation_id)
            if cached is not None and self._now() < cached[1] - _REFRESH_SKEW:
                return cached[0]
            token, expires_at = await self._mint(installation_id)
            self._cache[installation_id] = (token, expires_at)
            return token

    async def _mint(self, installation_id: int) -> Tuple[str, float]:
        response = await self.client.post(
            f"{_API}/app/installations/{int(installation_id)}/access_tokens",
            headers={
                **_API_HEADERS,
                "Authorization": f"Bearer {self._credentials.mint_jwt()}",
            },
        )
        if response.status_code != 201:
            raise GitHubConfigError(
                f"installation token request failed: HTTP {response.status_code}"
            )
        body = response.json()
        return body["token"], _parse_expiry(body["expires_at"])
