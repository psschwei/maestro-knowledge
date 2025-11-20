# Start with the slim base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory and environment variables
WORKDIR /app
ENV PYTHONPATH=/app
ENV UV_CACHE_DIR=/tmp/.uv-cache

# Runtime directories for OpenShift compatibility (writable locations)
ENV RUNTIME_DIR=/tmp
ENV LOG_DIR=/tmp
ENV PID_DIR=/tmp

# Install system packages and create the non-root user and app directory.
RUN apt-get update && \
    apt-get install -y --no-install-recommends procps && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash appuser --uid 1000 && \
    chown -R 1000:1000 /app

# Switch to the non-root user
USER appuser

# Copy the dependency definition file.
COPY --chown=1000:1000 pyproject.toml .

# Install Python dependencies.
RUN uv sync

# Copy the the maestro knowledge code.
COPY --chown=1000:1000 ./src /app/src
COPY --chown=1000:1000 start.sh stop.sh /app/
RUN chmod +x /app/start.sh

# Launch the server
EXPOSE 8030
ENTRYPOINT ["uv", "run", "./start.sh", "--host", "0.0.0.0", "--tail"]