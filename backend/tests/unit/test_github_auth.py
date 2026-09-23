"""Unit tests for the GitHub App auth and clone-env logic — no network.

The HTTP calls are replaced by a hand-rolled fake client (duck-typed on
``.post()`` / ``.get()``) rather than a library test transport, so these tests
do not depend on httpx2's mocking API. CLAUDE.md allows this here and only
here: the token/state/signing logic is what is under test, and none of it
stands in for proof the agent loop works.
"""
import base64
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import github.repo as repo_module  # noqa: E402
from github.auth import (  # noqa: E402
    AppCredentials,
    GitHubConfigError,
    InstallationNotFound,
    InstallationTokens,
    _parse_expiry,
    fetch_installation,
)
from github.repo import (  # noqa: E402
    _clone_argv,
    _clone_env,
    _redact,
    clone_repo,
    new_workspace,
    validate_repo,
)
from github.router import consume_state, issue_state  # noqa: E402


# ---------------------------------------------------------------- fakes

class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self):
        return self._body


class FakeClient:
    """Duck-typed stand-in for httpx2.AsyncClient."""

    def __init__(self, response=None, responder=None):
        self._response = response
        self._responder = responder
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self._responder:
            return self._responder(url, kwargs)
        return self._response

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self._responder:
            return self._responder(url, kwargs)
        return self._response


def _keypair():
    """A throwaway RSA keypair; the private half as an unencrypted PKCS#8 PEM."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


@pytest.fixture
def creds(tmp_path):
    private_pem, public_pem = _keypair()
    key_file = tmp_path / "github-app.pem"
    key_file.write_text(private_pem, encoding="utf-8")
    return AppCredentials(
        app_id="123456", private_key_path=str(key_file), slug="my-agent-app"
    ), public_pem


# ---------------------------------------------------------------- JWT

def test_jwt_is_signed_by_the_app_key(creds):
    credentials, public_pem = creds
    claims = pyjwt.decode(
        credentials.mint_jwt(), public_pem, algorithms=["RS256"]
    )
    assert claims["iss"] == "123456"  # a string: GitHub requires it, id is numeric
    assert isinstance(claims["iss"], str)


def test_jwt_lifetime_limits_and_backdated_iat(creds):
    credentials, public_pem = creds
    claims = pyjwt.decode(
        credentials.mint_jwt(), public_pem, algorithms=["RS256"],
        options={"verify_exp": False, "verify_iat": False},
    )
    # Backdated by ~60s. Asserted as an interval, not `iat < now`: int(now)
    # is already below a `time.time()` read a millisecond later, so the weak
    # form passes even with the backdate removed.
    assert 55 <= time.time() - claims["iat"] <= 65
    # GitHub's constraint is exp <= now + 600, not exp - iat <= 600. The
    # weaker form passes on a token sitting exactly on GitHub's limit.
    assert claims["exp"] - time.time() < 600
    assert claims["exp"] > time.time()


def test_jwt_rejected_by_a_foreign_key(creds):
    """Non-vacuous: the signature check is real, not just present."""
    credentials, _ = creds
    _, other_public = _keypair()
    with pytest.raises(pyjwt.InvalidSignatureError):
        pyjwt.decode(credentials.mint_jwt(), other_public, algorithms=["RS256"])


def test_missing_app_id_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(GitHubConfigError, match="GITHUB_APP_ID"):
        AppCredentials().app_id
    with pytest.raises(GitHubConfigError, match="GITHUB_APP_PRIVATE_KEY_PATH"):
        AppCredentials().private_key


def test_missing_key_file_raises(tmp_path):
    with pytest.raises(GitHubConfigError, match="not found"):
        AppCredentials(
            app_id="1", private_key_path=str(tmp_path / "nope.pem")
        ).private_key


def test_install_url_carries_state(creds):
    credentials, _ = creds
    assert credentials.install_url("abc") == (
        "https://github.com/apps/my-agent-app/installations/new?state=abc"
    )


# ------------------------------------------------------- installation tokens

def _expiry_from(now, seconds):
    from datetime import datetime, timedelta, timezone

    return (
        datetime.fromtimestamp(now + seconds, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


async def test_token_is_minted_then_cached(creds):
    credentials, _ = creds
    clock = [1_000_000.0]
    client = FakeClient(
        response=FakeResponse(
            201,
            {"token": "ghs_first", "expires_at": _expiry_from(clock[0], 3600)},
        )
    )
    tokens = InstallationTokens(
        credentials=credentials, client=client, now=lambda: clock[0]
    )

    assert await tokens.get(42) == "ghs_first"
    assert len(client.calls) == 1
    assert client.calls[0][1].endswith("/app/installations/42/access_tokens")
    assert client.calls[0][2]["headers"]["Authorization"].startswith("Bearer ")

    # Still well inside the hour: no second call.
    clock[0] += 600
    assert await tokens.get(42) == "ghs_first"
    assert len(client.calls) == 1


async def test_token_refreshes_inside_the_skew_window(creds):
    """A token with under five minutes left is replaced, not handed out."""
    credentials, _ = creds
    clock = [1_000_000.0]
    minted = []

    def responder(url, kwargs):
        minted.append(url)
        return FakeResponse(
            201,
            {
                "token": f"ghs_{len(minted)}",
                "expires_at": _expiry_from(clock[0], 3600),
            },
        )

    tokens = InstallationTokens(
        credentials=credentials, client=FakeClient(responder=responder),
        now=lambda: clock[0],
    )

    assert await tokens.get(7) == "ghs_1"
    # 3300s in: 300s of life left, exactly at the skew boundary -> refresh.
    clock[0] += 3300
    assert await tokens.get(7) == "ghs_2"
    assert len(minted) == 2


async def test_tokens_are_cached_per_installation(creds):
    credentials, _ = creds
    clock = [1_000_000.0]

    def responder(url, kwargs):
        return FakeResponse(
            201,
            {
                "token": "ghs_" + url.rsplit("/", 2)[-2],
                "expires_at": _expiry_from(clock[0], 3600),
            },
        )

    tokens = InstallationTokens(
        credentials=credentials, client=FakeClient(responder=responder),
        now=lambda: clock[0],
    )
    assert await tokens.get(1) == "ghs_1"
    assert await tokens.get(2) == "ghs_2"
    assert await tokens.get(1) == "ghs_1"  # from cache, still per-installation


async def test_token_failure_raises_without_the_body(creds):
    credentials, _ = creds
    client = FakeClient(response=FakeResponse(401, {"message": "Bad credentials"}))
    tokens = InstallationTokens(credentials=credentials, client=client)
    with pytest.raises(GitHubConfigError) as excinfo:
        await tokens.get(42)
    assert "HTTP 401" in str(excinfo.value)
    # The name of this test claims the body is not echoed; assert it, rather
    # than matching only the status code and leaving the claim untested.
    assert "Bad credentials" not in str(excinfo.value)


def test_expiry_parsing():
    # Pinned to the exact UTC epoch for 2026-09-21T12:00:00Z, so a parse that
    # treated the timestamp as local time fails wherever the tests run — the
    # aware-datetime form below is only a second opinion, and would coincide
    # with a naive parse under TZ=UTC.
    assert 1789992000.0 == pytest.approx(
        datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    )
    assert _parse_expiry("2026-09-21T12:00:00Z") == pytest.approx(1789992000.0)


def test_expiry_parsing_rejects_a_malformed_timestamp():
    with pytest.raises(ValueError):
        _parse_expiry("2026-09-21 12:00:00")


# ------------------------------------------------------- installation lookup

async def test_fetch_installation_rejects_a_foreign_id(creds):
    credentials, _ = creds
    client = FakeClient(response=FakeResponse(404, {"message": "Not Found"}))
    with pytest.raises(InstallationNotFound):
        await fetch_installation(credentials, 999, client=client)


async def test_fetch_installation_surfaces_other_errors(creds):
    credentials, _ = creds
    client = FakeClient(response=FakeResponse(500, {"message": "boom"}))
    with pytest.raises(GitHubConfigError, match="HTTP 500"):
        await fetch_installation(credentials, 999, client=client)


async def test_fetch_installation_returns_the_account(creds):
    credentials, _ = creds
    client = FakeClient(
        response=FakeResponse(
            200,
            {"id": 42, "account": {"login": "octocat", "type": "User"}},
        )
    )
    info = await fetch_installation(credentials, 42, client=client)
    assert info["account"]["login"] == "octocat"


# ---------------------------------------------------------------- repo rules

@pytest.mark.parametrize(
    "repo",
    [
        "../etc/passwd",
        "a/b/../../c",
        "--upload-pack=touch /tmp/x",
        "owner",
        "https://github.com/owner/name",
        "owner/name.git/../../../etc",
        "git@github.com:owner/name.git",
        "",
        "/owner/name",
        # Dot-only and trailing-dot components pass the character-class
        # pattern, and "a/.." as a clone destination is the workspace root
        # itself — which would collapse the sandbox's path containment.
        "a/..",
        "a/.",
        "a/...",
        "../name",
        "owner/name.",
        "owner/name..",
    ],
)
def test_validate_repo_rejects(repo):
    with pytest.raises(ValueError):
        validate_repo(repo)


@pytest.mark.parametrize(
    "repo", ["owner/name", "owner/name.js", "a-b_c.d/e-f_g.h", "o123/n456"]
)
def test_validate_repo_accepts(repo):
    assert validate_repo(repo) == repo


def test_clone_argv_shape():
    argv = _clone_argv("owner/name", Path("/tmp/x"))
    assert argv[:3] == ["git", "clone", "--depth"]
    assert argv[3] == "1"
    # `--` so a repository name can never be read as a flag.
    assert argv[4] == "--"
    assert argv[5] == "https://github.com/owner/name.git"


async def test_clone_hands_the_token_to_the_child_env_not_the_argv(tmp_path, monkeypatch):
    """The security invariant, asserted where it can actually fail.

    `_clone_argv` never receives the token, so checking *its* output for the
    token is unfalsifiable. `clone_repo` is where the token is an input, so
    this captures what it really hands the OS.
    """
    token = "ghs_secrettokenvalue"
    captured = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

        def kill(self):
            pass

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = kwargs.get("env")
        return FakeProcess()

    monkeypatch.setattr(repo_module.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(repo_module.shutil, "which", lambda name: "git")

    dest = await clone_repo(token, "owner/name", tmp_path)

    # Malformed clone: the token on the command line would land in the
    # sandbox's on-disk command log.
    assert not any(token in part for part in captured["argv"]), captured["argv"]
    assert captured["argv"][5] == "https://github.com/owner/name.git"
    # Carried in the environment of that one child instead.
    basic = captured["env"]["GIT_CONFIG_VALUE_0"].removeprefix(
        "Authorization: Basic "
    )
    assert base64.b64decode(basic).decode() == f"x-access-token:{token}"
    # <dest_parent>/<uuid>/<repo name>
    assert dest.parent.parent.resolve() == tmp_path.resolve()
    assert dest.name == "name"


def test_clone_env_carries_the_token_as_an_extraheader():
    token = "ghs_secrettokenvalue"
    env = _clone_env(token, base_env={"PATH": "/usr/bin", "GROQ_API_KEY": "sk-xyz"})

    basic = env["GIT_CONFIG_VALUE_0"].removeprefix("Authorization: Basic ")
    assert base64.b64decode(basic).decode() == f"x-access-token:{token}"
    assert env["GIT_CONFIG_KEY_0"] == "http.extraheader"
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_env_allow_list_drops_everything_else():
    """Fail closed: an unrelated secret in the environment is not inherited."""
    env = _clone_env("ghs_x", base_env={"PATH": "/usr/bin", "GROQ_API_KEY": "sk-xyz"})
    assert env["PATH"] == "/usr/bin"
    assert "GROQ_API_KEY" not in env


def test_redact_strips_the_token_from_error_text():
    assert _redact("fatal: could not read ghs_abc123", "ghs_abc123") == (
        "fatal: could not read ***"
    )
    assert _redact("no token here", "") == "no token here"


def test_redact_also_strips_the_base64_header_value():
    """The form that actually enters the child environment, not just the raw
    token — a path that makes git echo the header would surface this one."""
    token = "ghs_abc123"
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    assert basic not in _redact(f"Sent header: {basic}", token)
    assert token not in _redact(f"Sent header: {basic}", token)


async def test_aclose_lets_the_client_be_reused(creds):
    """Closing must clear the cached client, not leave a dead one behind."""
    credentials, _ = creds

    class CountingClient(FakeClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.closed = False

        async def aclose(self):
            self.closed = True

    client = CountingClient(
        response=FakeResponse(
            201,
            {
                "token": "ghs_one",
                "expires_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            },
        )
    )
    tokens = InstallationTokens(credentials=credentials, client=client)
    await tokens.aclose()
    assert client.closed is True
    assert tokens._client is None, (
        "the closed client was left cached; every later get() would raise"
    )


def test_new_workspace_is_fresh_and_unique(tmp_path):
    first = new_workspace(tmp_path)
    second = new_workspace(tmp_path)
    assert first.is_dir() and second.is_dir()
    assert first != second


# ---------------------------------------------------------------- state

def test_state_is_single_use():
    state = issue_state()
    assert consume_state(state) is True
    assert consume_state(state) is False


def test_state_expires():
    expired = issue_state(now=100.0)
    assert consume_state(expired, now=100.0 + 601) is False

    fresh = issue_state(now=100.0)
    assert consume_state(fresh, now=100.0 + 599) is True


def test_unknown_state_is_rejected():
    assert consume_state("never-issued") is False


def test_issue_state_sweeps_expired_entries():
    import github.router as router

    router._pending_states.clear()
    issue_state(now=0.0)
    issue_state(now=1000.0)  # past the first one's deadline
    assert len(router._pending_states) == 1
    router._pending_states.clear()
