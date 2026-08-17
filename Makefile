AGENT_WASM := crates/tds-agent/target/wasm32-unknown-unknown/release/tds_agent.wasm
POLICY     ?= policy/mock.toml
TASK       ?= What time is it?\n!tool now {}
IMAGE      ?= tds-wasm-ai:dev

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: agent
agent: $(AGENT_WASM) ## Compile the agent to WebAssembly

$(AGENT_WASM): $(shell find crates/tds-agent/src crates/tds-abi/src -name '*.rs') crates/tds-agent/Cargo.toml
	cd crates/tds-agent && cargo build --release --target wasm32-unknown-unknown
	@ls -lh $(AGENT_WASM)

.PHONY: host
host: ## Build the host runtime
	cargo build --release -p tds-host

.PHONY: build
build: agent host ## Build everything

.PHONY: test
test: agent ## Run the full test suite
	cargo test --workspace

.PHONY: lint
lint: ## Clippy and rustfmt checks
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --check
	cd crates/tds-agent && cargo fmt --check

.PHONY: fmt
fmt: ## Format all crates
	cargo fmt
	cd crates/tds-agent && cargo fmt

.PHONY: check
check: agent host ## Validate the policy and the agent module (no credential needed)
	./target/release/tds-host check --offline --agent $(AGENT_WASM) --policy $(POLICY)

.PHONY: run
run: agent host ## Run one task (TASK=... POLICY=...)
	./target/release/tds-host run --agent $(AGENT_WASM) --policy $(POLICY) --task "$$(printf '$(TASK)')"

.PHONY: serve
serve: agent host ## Serve the agent on :8080
	./target/release/tds-host serve --agent $(AGENT_WASM) --policy $(POLICY)

.PHONY: docker
docker: ## Build the container image
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run: docker ## Run the container with the offline policy
	docker run --rm -p 8080:8080 \
		--read-only --cap-drop ALL --security-opt no-new-privileges \
		-e TDS_POLICY=/app/policy/mock.toml \
		$(IMAGE) serve

.PHONY: clean
clean: ## Remove build artifacts
	cargo clean
	cd crates/tds-agent && cargo clean
