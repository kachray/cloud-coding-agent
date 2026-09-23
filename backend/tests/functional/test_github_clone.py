"""Functional tests for the GitHub App clone path — real GitHub, real clone.

Requires a GitHub App installed on a PRIVATE scratch repo, with these in
backend/.env:

    GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_PATH, GITHUB_APP_SLUG,
    GITHUB_TEST_INSTALLATION_ID, GITHUB_TEST_REPO

They skip when those are absent. A skip is honest where a mock would not be:
the tests below are real and simply cannot run without an App, so the milestone
is not verified until they execute rather than skip.
"""
import os
import sys
from pathlib import Path

import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.loop import AgentLoop  # noqa: E402
from github import GitHubClient, InstallationTokens, clone_repo  # noqa: E402

_REQUIRED = (
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY_PATH",
    "GITHUB_TEST_INSTALLATION_ID",
    "GITHUB_TEST_REPO",
)

requires_app = pytest.mark.skipif(
    any(not os.environ.get(name) for name in _REQUIRED),
    reason="GitHub App not configured; set " + ", ".join(_REQUIRED) + " in .env",
)

pytestmark = requires_app


@pytest.fixture
def installation_id() -> int:
    return int(os.environ["GITHUB_TEST_INSTALLATION_ID"])


@pytest.fixture
def repo() -> str:
    return os.environ["GITHUB_TEST_REPO"]


@pytest.fixture
async def tokens():
    provider = InstallationTokens()
    yield provider
    if provider._client is not None:
        await provider._client.aclose()


def _git(dest: Path, *args: str) -> str:
    """Run a local git query in an existing clone (no network, no token)."""
    import subprocess

    return subprocess.run(
        ["git", *args],
        cwd=dest,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


async def _remote_head_sha(repo: str, token: str) -> str:
    """GitHub's own view of the repository head, fetched over the API.

    Independent of the clone, so a stale or partial clone cannot satisfy it.
    """
    async with httpx2.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"https://api.github.com/repos/{repo}/commits/HEAD",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        assert response.status_code == 200, (
            f"could not read {repo} over the API: HTTP {response.status_code}"
        )
        return response.json()["sha"]


def _files_containing(root: Path, needle: str) -> list:
    """Every file under *root* whose text contains *needle*."""
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(path))
        except OSError:
            continue
    return hits


class TestInstallationTokenClone:

    async def test_clone_private_repo_with_installation_token(
        self, tokens, repo, tmp_path, installation_id
    ):
        token = await tokens.get(installation_id)
        assert token, "no installation token was minted"

        dest = await clone_repo(token, repo, tmp_path)

        assert (dest / ".git").is_dir(), f"no .git in the clone at {dest}"
        local_head = _git(dest, "rev-parse", "HEAD")
        assert local_head == await _remote_head_sha(repo, token), (
            "cloned HEAD does not match GitHub's HEAD for this repository"
        )

    async def test_clone_without_a_token_fails(self, repo, tmp_path):
        """The non-vacuity check for the test above.

        The scratch repo is private, so a successful clone there is proof the
        token did the work. Without this test, a repo that had quietly become
        public would let the whole suite pass on anonymous access.
        """
        with pytest.raises(RuntimeError, match="git clone"):
            await clone_repo("", repo, tmp_path)

    async def test_token_never_reaches_disk(
        self, tokens, repo, tmp_path, installation_id
    ):
        token = await tokens.get(installation_id)
        dest = await clone_repo(token, repo, tmp_path)

        # The URL-embedded-token approach would put this in the clone's config.
        config = (dest / ".git" / "config").read_text(encoding="utf-8")
        assert "x-access-token" not in config
        assert token not in config

        # Nor anywhere else the clone writes.
        assert _files_containing(tmp_path, token) == [], (
            "the installation token was written to disk under the workspace"
        )
        assert _files_containing(tmp_path, "ghs_") == []

        # No command-log assertion here on purpose: this test calls
        # clone_repo directly, which never touches the sandbox, so no log
        # could exist regardless of how the clone was implemented. The
        # token-in-the-log case is asserted in the agent test below, which is
        # the only one that has a sandbox to log to.


class TestAgentClonesAndWorksInRepo:

    async def test_agent_clones_repo_then_reads_it(
        self, sandbox, client, repo, tmp_path, installation_id
    ):
        agent = AgentLoop(
            sandbox=sandbox,
            model="openai/gpt-oss-120b",
            client=client,
            github=GitHubClient(
                installation_id=installation_id, workspaces_root=tmp_path / "ws"
            ),
            system_instruction=(
                "You are a coding agent operating in the given working "
                "directory. Use the provided tools to complete the user's task "
                "exactly. When finished, briefly state the outcome."
            ),
        )

        result = await agent.run(
            f"Clone the repository {repo} with the github_clone tool, then "
            f"list the files in it and report what you find.",
            working_dir=tmp_path,
        )

        names = [c["name"] for c in agent.tool_calls]
        assert "github_clone" in names, (
            f"the agent never called github_clone. Loop result:\n{result}"
        )

        # The clone exists on disk, and the session's working directory moved
        # into it — that is what makes the file and shell tools operate inside
        # the repository.
        working_dir = Path(sandbox.working_dir)
        assert (working_dir / ".git").is_dir(), (
            f"sandbox.working_dir {working_dir} is not a git clone"
        )
        assert working_dir.is_relative_to(tmp_path / "ws")

        # And the agent actually used the clone afterwards. This is the
        # assertion that makes the test about a usable loop rather than about
        # a clone that happened: a run that clones and then reports nothing
        # would pass everything above.
        clone_at = names.index("github_clone")
        assert any(
            name in ("read_file", "run_in_shell") for name in names[clone_at + 1:]
        ), (
            f"the agent never read or listed anything after cloning. "
            f"Tool calls were: {names}"
        )
        assert result and result.strip(), "the agent produced no final output"
        assert list(working_dir.glob("README*")), (
            f"the clone at {working_dir} has no README, so the agent cannot "
            f"have read one"
        )
        # Note: the README's text is deliberately not asserted against
        # `result`. CLAUDE.md's Risk 2 is silent single-character corruption
        # when the model transcribes a supplied exact string, so an assertion
        # on verbatim transcription would be a flaky test of the model rather
        # than of the clone path.

        # The token must not have reached the model's context, in any field.
        for call in agent.tool_calls:
            for field in ("args", "result_preview", "error"):
                assert "ghs_" not in str(call.get(field, "")), (
                    f"a token reached the model's context: {call}"
                )

        # Nor the sandbox's on-disk audit trail, which is where an
        # argv-carried token would have landed.
        log = sandbox.command_log_path
        if log.exists():
            assert "ghs_" not in log.read_text(encoding="utf-8", errors="ignore")
