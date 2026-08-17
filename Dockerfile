# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: compile the agent to WebAssembly and the host to a native binary.
# ---------------------------------------------------------------------------
FROM rust:1-bookworm AS builder

RUN rustup target add wasm32-unknown-unknown

WORKDIR /src
COPY . .

# Cache mounts keep the registry and target directories across builds. The
# artifacts are copied out of the cache in the same layer, since a cache mount
# is not part of the image.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    --mount=type=cache,target=/src/crates/tds-agent/target \
    set -eux; \
    cargo build --release -p tds-host; \
    cd crates/tds-agent && cargo build --release --target wasm32-unknown-unknown; \
    cd /src; \
    mkdir -p /out; \
    cp target/release/tds-host /out/tds-host; \
    cp crates/tds-agent/target/wasm32-unknown-unknown/release/tds_agent.wasm /out/agent.wasm

# ---------------------------------------------------------------------------
# Runtime stage: the host binary, the agent module, and a CA bundle. Nothing else.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin tds

WORKDIR /app
COPY --from=builder /out/tds-host /usr/local/bin/tds-host
COPY --from=builder /out/agent.wasm /app/agent.wasm
COPY policy/ /app/policy/

# The agent module and policy are read-only to the process that loads them, and
# the container needs no writable filesystem at all — run it with
# `--read-only` and the guarantee holds from the outside too.
RUN chmod -R a-w /app

USER 10001:10001

ENV TDS_AGENT=/app/agent.wasm \
    TDS_POLICY=/app/policy/default.toml \
    TDS_ADDR=0.0.0.0:8080 \
    TDS_LOG=info

EXPOSE 8080

# `check` loads the policy and the module without running the agent, so an
# unhealthy container is one that could not serve a request anyway.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/tds-host", "check"]

ENTRYPOINT ["/usr/local/bin/tds-host"]
CMD ["serve"]
