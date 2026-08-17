//! `tds-host` — a containerized WebAssembly runtime that encapsulates an AI agent.
//!
//! Two ways in: `run` for a single task, `serve` for an HTTP endpoint. Both put
//! the same sandbox around the same agent module.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tds_abi::AgentInput;

use tds_host::policy::Policy;
use tds_host::runtime::{RunReport, Runtime};

#[derive(Parser)]
#[command(
    name = "tds-host",
    version,
    about = "Run a WebAssembly-sandboxed AI agent with host-mediated capabilities."
)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct CommonArgs {
    /// Path to the agent's compiled `.wasm` module.
    #[arg(long, env = "TDS_AGENT", default_value = "agent.wasm", global = true)]
    agent: PathBuf,

    /// Path to the sandbox policy. Omit to use the built-in defaults.
    #[arg(long, env = "TDS_POLICY", global = true)]
    policy: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a single task and print the result.
    Run {
        /// The task for the agent.
        #[arg(long)]
        task: String,

        /// Extra context prepended to the conversation.
        #[arg(long)]
        context: Option<String>,

        /// Model turns to allow. Clamped to the policy's ceiling.
        #[arg(long, default_value_t = 4)]
        max_steps: u32,

        /// Emit the full report as JSON instead of a human-readable summary.
        #[arg(long)]
        json: bool,
    },

    /// Serve the agent over HTTP.
    Serve {
        #[arg(long, env = "TDS_ADDR", default_value = "0.0.0.0:8080")]
        addr: SocketAddr,
    },

    /// Validate the policy and the agent module, then exit. Checks the model
    /// credential too, so it doubles as a container healthcheck.
    Check {
        /// Skip the credential check, so a policy can be validated without one.
        #[arg(long)]
        offline: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing();

    let policy = match &cli.common.policy {
        Some(path) => Policy::load(path)?,
        None => {
            let policy = Policy::default();
            policy.validate()?;
            policy
        }
    };

    match cli.command {
        Command::Check { offline } => {
            let runtime = if offline {
                // Everything except the credential: parse the policy, compile the
                // module, and verify its imports and exports.
                let wasm = std::fs::read(&cli.common.agent).with_context(|| {
                    format!("reading agent module {}", cli.common.agent.display())
                })?;
                Runtime::new(policy, &wasm, Box::new(tds_host::llm::MockProvider))?
            } else {
                Runtime::from_files(policy, &cli.common.agent)?
            };
            println!(
                "ok: policy valid, agent module accepted ({} tool(s) enabled{})",
                runtime.policy().tools.enabled.len(),
                if offline {
                    ", credential not checked"
                } else {
                    ""
                }
            );
            Ok(())
        }

        Command::Run {
            task,
            context,
            max_steps,
            json,
        } => {
            let runtime = Runtime::from_files(policy, &cli.common.agent)?;
            let report = runtime.run(AgentInput {
                task,
                context,
                max_steps,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_report(&report);
            }
            Ok(())
        }

        Command::Serve { addr } => {
            let runtime = Arc::new(Runtime::from_files(policy, &cli.common.agent)?);
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("starting the async runtime")?
                .block_on(serve(runtime, addr))
        }
    }
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_env("TDS_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = fmt().with_env_filter(filter).with_target(true);
    if std::env::var("TDS_LOG_FORMAT").as_deref() == Ok("json") {
        builder.json().init();
    } else {
        builder.init();
    }
}

fn print_report(report: &RunReport) {
    println!("{}", report.output.answer);
    println!();
    println!(
        "— {:?} after {} step(s) in {} ms via {} (fuel {}, tokens {}/{})",
        report.output.stop,
        report.output.steps,
        report.duration_ms,
        report.model,
        report.fuel_used,
        report.output.usage.input_tokens,
        report.output.usage.output_tokens,
    );
    for call in &report.output.tool_calls {
        let mark = if call.ok { "ok" } else { "failed" };
        println!("  step {} · {} · {mark}", call.step, call.name);
    }
}

// ---------------------------------------------------------------------------
// HTTP surface
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RunRequest {
    task: String,
    #[serde(default)]
    context: Option<String>,
    #[serde(default = "default_steps")]
    max_steps: u32,
}

fn default_steps() -> u32 {
    4
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

async fn serve(runtime: Arc<Runtime>, addr: SocketAddr) -> Result<()> {
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/policy", get(policy_handler))
        .route("/v1/runs", post(run_handler))
        .with_state(runtime);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serving")?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}

/// The effective policy, so an operator can confirm what the agent can reach.
/// Only the *name* of the credential variable appears; the credential does not.
async fn policy_handler(State(runtime): State<Arc<Runtime>>) -> Json<serde_json::Value> {
    let policy = runtime.policy();
    Json(serde_json::json!({
        "limits": policy.limits,
        "llm": {
            "provider": policy.llm.provider,
            "model": policy.llm.model,
            "effort": policy.llm.effort,
            "thinking": policy.llm.thinking,
            "max_tokens": policy.llm.max_tokens,
            "api_key_env": policy.llm.api_key_env,
        },
        "tools": {
            "enabled": policy.tools.enabled,
            "http_allowed_hosts": policy.tools.http.allowed_hosts,
        }
    }))
}

async fn run_handler(
    State(runtime): State<Arc<Runtime>>,
    Json(request): Json<RunRequest>,
) -> Result<Json<RunReport>, (StatusCode, Json<ErrorBody>)> {
    if request.task.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: "task must not be empty".into(),
            }),
        ));
    }

    let input = AgentInput {
        task: request.task,
        context: request.context,
        max_steps: request.max_steps,
    };

    // Running the agent blocks: Wasmtime execution and the model call are both
    // synchronous, so they belong off the async worker threads.
    let report = tokio::task::spawn_blocking(move || runtime.run(input))
        .await
        .map_err(|e| internal(format!("run task panicked: {e}")))?
        .map_err(|e| internal(format!("{e:#}")))?;

    Ok(Json(report))
}

fn internal(message: String) -> (StatusCode, Json<ErrorBody>) {
    tracing::error!("{message}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorBody { error: message }),
    )
}
