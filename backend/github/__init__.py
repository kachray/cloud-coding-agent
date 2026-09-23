"""GitHub App integration: install flow, installation tokens, cloning.

Deliberately named for the service rather than for a library. It shadows the
unmaintained PyPI ``github`` package; nothing here or in this project's
dependency tree imports that, so the name is free. If a library with a claim on
it is ever added, this package is renamed.
"""
from .auth import (
    AppCredentials,
    GitHubConfigError,
    InstallationNotFound,
    InstallationTokens,
    fetch_installation,
)
from .repo import GitHubClient, clone_repo, new_workspace, validate_repo

__all__ = [
    "AppCredentials",
    "GitHubClient",
    "GitHubConfigError",
    "InstallationNotFound",
    "InstallationTokens",
    "clone_repo",
    "fetch_installation",
    "new_workspace",
    "validate_repo",
]
