FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
--mount=type=bind,source=uv.lock,target=uv.lock \
--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
uv sync --frozen --no-install-project --no-dev --no-editable

# fetch docs before adding the rest of the source code
RUN mkdir /opt/groundlight
COPY fetch-docs.sh /app/fetch-docs.sh
RUN bash -ex /app/fetch-docs.sh

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
uv sync --frozen --no-dev --no-editable
RUN cp -r /app/resources /opt/groundlight/docs/resources

FROM python:3.11-slim-bookworm

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=uv --chown=app:app /app/.venv /app/.venv
COPY --from=uv --chown=app:app /opt/groundlight /opt/groundlight

ENV PATH="/app/.venv/bin:$PATH"

RUN ls -la /app/.venv/bin

ENTRYPOINT ["groundlight-mcp-server"]
