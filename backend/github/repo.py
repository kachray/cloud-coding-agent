"""Repository validation, workspace allocation, and token-safe cloning.

The clone runs as its own subprocess rather than through
``SandboxInterface.run_in_shell``, and that is a security decision, not a
layering preference: the sandbox appends every command it runs to
``<workspace>.commands.log`` on disk, so a clone command carrying a credential
would put that credential in the agent's audit log. Here the token never
reaches the command line at all.
"""
import asyncio
import base64
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from .auth import InstallationTokens

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACES_ROOT = BACKEND_ROOT / "workspaces"

# A clone is a network operation against a possibly large repository, so it
# gets far more than the sandbox's 30s default command timeout.
_CLONE_TIMEOUT = 300.0

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

# Built by allow-list rather than by copying os.environ and deleting a few
# names. An allow-list fails closed: a secret added to the environment later is
# not silently handed to git. (The sandbox's own shells pass no `env=` and
# inherit everything — see CLAUDE.md's note on the containment that isn't
# there.)
_CLONE_ENV_KEEP = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SystemRoot",
    "SystemDrive",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "LANG",
)


def validate_repo(repo: str) -> str:
    """Require a bare ``owner/name`` and reject everything else.

    The model supplies this string. A value it gets wrong should come back as a
    correctable tool error rather than as arguments to git — this is the check
    that keeps a leading ``-`` from being read as a flag, and keeps a URL or a
    path from being passed where a repository name belongs.
    """
    if not isinstance(repo, str) or not _REPO_PATTERN.match(repo):
        raise ValueError(
            f"invalid repository {repo!r}: expected 'owner/name' using only "
            f"letters, digits, '.', '_' and '-'"
        )
    owner, name = repo.split("/")
    # The pattern above permits dot-only components, and ".." as the name is
    # not harmless: `new_workspace(root) / ".."` resolves to `root` itself, so
    # the session's working_dir would become the shared workspaces directory
    # and `_resolve_inside`'s containment would collapse to it, exposing every
    # other session's clone. A trailing dot also cannot round-trip through
    # Windows path handling.
    if owner.endswith(".") or name.endswith("."):
        raise ValueError(
            f"invalid repository {repo!r}: owner and name cannot be empty, "
            f"dot-only, or end with '.'"
        )
    return repo


def new_workspace(root: Union[str, Path]) -> Path:
    """Create and return a fresh, unique session directory under *root*.

    The clone lands one level below this, so the session directory stays free
    for things that must not be inside the repository.
    """
    path = Path(root) / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=False)
    return path


def _clone_env(token: str, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment for one ``git`` child process, carrying the token.

    ``http.extraheader`` is how a token that is not a password authenticates:
    git would otherwise prompt for one. Passing it through ``GIT_CONFIG_*``
    instead of ``-c`` keeps it off the command line, so it lands in neither the
    process list nor the sandbox's command log.

    What is deliberately *not* done: embedding the token in the clone URL. That
    is the obvious approach and it writes
    ``https://x-access-token:ghs_…@github.com/…`` into the clone's
    ``.git/config``, where it sits for the life of the workspace.
    """
    source = os.environ if base_env is None else base_env
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = {key: source[key] for key in _CLONE_ENV_KEEP if key in source}
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
            # A genuine auth failure must fail immediately rather than hang
            # until the timeout on a prompt nobody can answer.
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def _clone_argv(repo: str, dest: Path) -> List[str]:
    return [
        "git",
        "clone",
        "--depth",
        "1",
        "--",
        f"https://github.com/{repo}.git",
        str(dest),
    ]


def _redact(text: str, token: str) -> str:
    """Strip the credential from anything on its way back to the model.

    Both forms are removed: the raw token, and the base64
    ``x-access-token:<token>`` value that is what actually enters the child
    environment. Only the raw form appears today, but a path that makes git
    echo the header (a verbose-curl flag, a proxy error) would surface the
    base64 one, and this is the only choke point between git's output and the
    model's context.
    """
    if not token:
        return text
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return text.replace(token, "***").replace(basic, "***")


async def clone_repo(
    token: str,
    repo: str,
    dest_parent: Union[str, Path],
) -> Path:
    """Clone *repo* into a fresh workspace under *dest_parent*; return the path."""
    repo = validate_repo(repo)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "git is not on PATH; cannot clone. Install git and retry."
        )

    dest = new_workspace(dest_parent) / repo.split("/")[1]
    process = await asyncio.create_subprocess_exec(
        *_clone_argv(repo, dest),
        env=_clone_env(token),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=_CLONE_TIMEOUT
        )
    except asyncio.TimeoutError:
        process.kill()
        raise RuntimeError(
            f"git clone {repo} timed out after {_CLONE_TIMEOUT:.0f}s"
        ) from None

    if process.returncode != 0:
        detail = _redact(
            stderr.decode(errors="replace").strip()[-500:], token
        )
        raise RuntimeError(
            f"git clone {repo} failed (exit {process.returncode}): {detail}"
        )
    return dest


class GitHubClient:
    """The GitHub surface the agent loop holds.

    Binding the installation id here rather than exposing it as a tool
    parameter is deliberate. The model has no reason to handle that number and
    every reason not to: it would have to transcribe it into a tool call, which
    is exactly the silent single-character corruption CLAUDE.md's Risk 2
    describes.
    """

    def __init__(
        self,
        installation_id: int,
        tokens: Optional[InstallationTokens] = None,
        workspaces_root: Union[str, Path] = DEFAULT_WORKSPACES_ROOT,
    ) -> None:
        self.installation_id = int(installation_id)
        self._tokens = tokens or InstallationTokens()
        self.workspaces_root = Path(workspaces_root)

    async def clone(self, repo: str) -> Path:
        """Clone *repo* into a fresh workspace; return the clone path.

        The token is fetched here, immediately before the clone, so a session
        that has been running for hours gets a fresh one rather than one that
        expired while it was idle.
        """
        token = await self._tokens.get(self.installation_id)
        return await clone_repo(token, repo, self.workspaces_root)

    async def aclose(self) -> None:
        """Release the underlying HTTP client (connection pool)."""
        await self._tokens.aclose()
