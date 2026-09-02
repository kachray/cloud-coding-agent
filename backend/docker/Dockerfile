FROM python:3.12-slim
# Dev-tools image for the agent sandbox: python + node + git + ripgrep + jq +
# compilers covers "arbitrary project commands" without a 2GB devcontainer.
# ponytail: Debian bookworm's nodejs is v18 — add a NodeSource layer if a task
# needs a modern Node.
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash git curl ripgrep jq nodejs npm build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
