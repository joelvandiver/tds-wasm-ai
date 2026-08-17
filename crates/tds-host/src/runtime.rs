//! The sandbox: a Wasmtime engine, a policy, and three host functions.
//!
//! The guest module is instantiated fresh for every run, so each run gets its
//! own linear memory, its own fuel budget, and its own tool state. Nothing
//! carries over between runs but the compiled code.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use tds_abi::{
    cap, error_code, unpack_ptr_len, AgentInput, AgentOutput, CapError, CapResponse, LlmRequest,
    LlmResponse, LogLevel, ToolInvokeRequest, ToolInvokeResponse, ToolsListResponse, ABI_VERSION,
    HOST_MODULE,
};
use wasmtime::{
    Caller, Config, Engine, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, Trap,
};

use crate::llm::LlmProvider;
use crate::policy::Policy;
use crate::tools::ToolBox;

/// How often the epoch counter advances. The wall-clock budget is expressed in
/// these ticks, so it is the granularity of interruption.
const EPOCH_TICK: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Serialize)]
pub struct LogLine {
    pub level: String,
    pub message: String,
}

/// Everything observable about one agent run.
#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    pub output: AgentOutput,
    pub model: String,
    pub duration_ms: u128,
    pub fuel_used: u64,
    pub logs: Vec<LogLine>,
}

/// State the host keeps for a single run. This is the store data, so every host
/// function reaches it through its `Caller`.
struct HostState {
    policy: Arc<Policy>,
    llm: Arc<dyn LlmProvider>,
    tools: ToolBox,
    /// The response to the most recent capability call, waiting to be read.
    pending: Vec<u8>,
    logs: Vec<LogLine>,
    llm_calls: u32,
    limits: StoreLimits,
}

pub struct Runtime {
    engine: Engine,
    module: Module,
    policy: Arc<Policy>,
    llm: Arc<dyn LlmProvider>,
    ticker_stop: Arc<AtomicBool>,
    ticker: Option<JoinHandle<()>>,
}

impl Runtime {
    pub fn new(policy: Policy, wasm: &[u8], llm: Box<dyn LlmProvider>) -> Result<Runtime> {
        policy.validate()?;

        let mut config = Config::new();
        config.consume_fuel(true);
        config.epoch_interruption(true);
        config.wasm_backtrace(true);

        let engine = Engine::new(&config).context("creating the Wasm engine")?;
        let module = Module::new(&engine, wasm).context("compiling the agent module")?;

        verify_module_shape(&module)?;

        // Advance the epoch on a background thread. This is what lets the host
        // interrupt a guest that has stopped cooperating — an infinite loop with
        // no host calls in it still gets stopped.
        let ticker_stop = Arc::new(AtomicBool::new(false));
        let ticker = {
            let engine = engine.clone();
            let stop = ticker_stop.clone();
            std::thread::Builder::new()
                .name("tds-epoch".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(EPOCH_TICK);
                        engine.increment_epoch();
                    }
                })
                .context("spawning the epoch ticker")?
        };

        Ok(Runtime {
            engine,
            module,
            policy: Arc::new(policy),
            llm: Arc::from(llm),
            ticker_stop,
            ticker: Some(ticker),
        })
    }

    pub fn from_files(policy: Policy, wasm_path: &std::path::Path) -> Result<Runtime> {
        let wasm = std::fs::read(wasm_path)
            .with_context(|| format!("reading agent module {}", wasm_path.display()))?;
        let llm = crate::llm::build(&policy.llm)?;
        Runtime::new(policy, &wasm, llm)
    }

    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Run the agent once, in a fresh instance.
    pub fn run(&self, mut input: AgentInput) -> Result<RunReport> {
        // The guest may ask for fewer steps than the policy allows, never more.
        input.max_steps = input.max_steps.clamp(1, self.policy.limits.max_steps);

        let started = Instant::now();
        let limits = self.policy.limits.clone();

        let state = HostState {
            policy: self.policy.clone(),
            llm: self.llm.clone(),
            tools: ToolBox::new(self.policy.clone())?,
            pending: Vec::new(),
            logs: Vec::new(),
            llm_calls: 0,
            limits: StoreLimitsBuilder::new()
                .memory_size(limits.memory_bytes)
                .table_elements(limits.table_elements)
                .instances(1)
                .memories(1)
                .tables(1)
                .build(),
        };

        let mut store = Store::new(&self.engine, state);
        store.limiter(|s| &mut s.limits);
        store
            .set_fuel(limits.fuel)
            .context("setting the fuel budget")?;
        // One deadline for the whole run. A guest that loops forever, and a guest
        // that waits forever on a slow model, are both stopped by it.
        let ticks = (limits.wall_clock_ms / EPOCH_TICK.as_millis() as u64).max(1);
        store.set_epoch_deadline(ticks);

        let mut linker: Linker<HostState> = Linker::new(&self.engine);
        register_host_functions(&mut linker)?;

        let instance = linker
            .instantiate(&mut store, &self.module)
            .context("instantiating the agent module")?;

        check_abi_version(&mut store, &instance)?;

        let result = invoke_agent(&mut store, &instance, &input);

        let fuel_used = limits.fuel.saturating_sub(store.get_fuel().unwrap_or(0));
        let logs = std::mem::take(&mut store.data_mut().logs);

        let output = match result {
            Ok(output) => output,
            Err(err) => {
                // Turn a trap into an outcome rather than an opaque failure: the
                // caller wants to know the agent was stopped and why.
                let reason = describe_trap(&err);
                tracing::warn!(error = %err, "agent run did not complete");
                AgentOutput {
                    answer: reason,
                    steps: 0,
                    stop: tds_abi::StopReason::Error,
                    tool_calls: Vec::new(),
                    usage: Default::default(),
                }
            }
        };

        Ok(RunReport {
            output,
            model: self.llm.describe(),
            duration_ms: started.elapsed().as_millis(),
            fuel_used,
            logs,
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.ticker_stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.ticker.take() {
            let _ = handle.join();
        }
    }
}

/// Reject a module that imports anything beyond the host ABI. This is the check
/// that makes "no ambient authority" a property of the runtime rather than a
/// promise about how the guest was compiled — a module built against WASI, or
/// against some other host, is refused here instead of at first use.
fn verify_module_shape(module: &Module) -> Result<()> {
    let allowed = ["log", "call", "response_read"];
    for import in module.imports() {
        if import.module() != HOST_MODULE || !allowed.contains(&import.name()) {
            bail!(
                "agent module imports {}::{}, which this host does not provide; \
                 the agent may only import {HOST_MODULE}::{{log, call, response_read}}",
                import.module(),
                import.name()
            );
        }
    }

    for required in [
        "memory",
        "tds_abi_version",
        "tds_alloc",
        "tds_dealloc",
        "tds_run",
    ] {
        if module.get_export(required).is_none() {
            bail!("agent module does not export {required:?}");
        }
    }
    Ok(())
}

fn check_abi_version(store: &mut Store<HostState>, instance: &wasmtime::Instance) -> Result<()> {
    let version = instance
        .get_typed_func::<(), i32>(&mut *store, "tds_abi_version")
        .context("agent module has no usable tds_abi_version export")?
        .call(&mut *store, ())
        .context("calling tds_abi_version")?;
    if version != ABI_VERSION {
        bail!("agent module speaks ABI version {version}, host speaks {ABI_VERSION}");
    }
    Ok(())
}

fn invoke_agent(
    store: &mut Store<HostState>,
    instance: &wasmtime::Instance,
    input: &AgentInput,
) -> Result<AgentOutput> {
    let memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or_else(|| anyhow!("agent module does not export its memory"))?;
    let alloc = instance.get_typed_func::<u32, u32>(&mut *store, "tds_alloc")?;
    let dealloc = instance.get_typed_func::<(u32, u32), ()>(&mut *store, "tds_dealloc")?;
    let run = instance.get_typed_func::<(u32, u32), u64>(&mut *store, "tds_run")?;

    let encoded = serde_json::to_vec(input)?;
    let len = encoded.len() as u32;
    let ptr = alloc.call(&mut *store, len)?;
    if ptr == 0 {
        bail!("agent module could not allocate {len} bytes for its input");
    }
    memory
        .write(&mut *store, ptr as usize, &encoded)
        .context("writing agent input into guest memory")?;

    let packed = run.call(&mut *store, (ptr, len))?;
    dealloc.call(&mut *store, (ptr, len))?;

    let (out_ptr, out_len) = unpack_ptr_len(packed);
    if out_ptr == 0 || out_len == 0 {
        bail!("agent returned an empty result");
    }

    let mut buffer = vec![0u8; out_len as usize];
    memory
        .read(&mut *store, out_ptr as usize, &mut buffer)
        .context("reading the agent result from guest memory")?;
    dealloc.call(&mut *store, (out_ptr, out_len))?;

    serde_json::from_slice(&buffer).context("decoding the agent result")
}

fn describe_trap(err: &anyhow::Error) -> String {
    match err.downcast_ref::<Trap>() {
        Some(Trap::OutOfFuel) => {
            "The agent was stopped: it exhausted its compute budget (limits.fuel).".to_string()
        }
        Some(Trap::Interrupt) => {
            "The agent was stopped: it exceeded its wall-clock budget (limits.wall_clock_ms)."
                .to_string()
        }
        Some(Trap::UnreachableCodeReached) => {
            "The agent panicked inside the sandbox and was stopped.".to_string()
        }
        Some(other) => format!("The agent trapped and was stopped: {other}."),
        None => format!("The agent run failed: {err:#}"),
    }
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

fn register_host_functions(linker: &mut Linker<HostState>) -> Result<()> {
    linker.func_wrap(
        HOST_MODULE,
        "log",
        |mut caller: Caller<'_, HostState>, level: i32, ptr: u32, len: u32| {
            // A log line is bounded so a chatty guest cannot exhaust host memory.
            let len = len.min(8 * 1024);
            let Some(bytes) = read_guest(&mut caller, ptr, len) else {
                return;
            };
            let level = LogLevel::from_i32(level);
            let message = String::from_utf8_lossy(&bytes).into_owned();
            match level {
                LogLevel::Debug => tracing::debug!(target: "agent", "{message}"),
                LogLevel::Info => tracing::info!(target: "agent", "{message}"),
                LogLevel::Warn => tracing::warn!(target: "agent", "{message}"),
                LogLevel::Error => tracing::error!(target: "agent", "{message}"),
            }
            let logs = &mut caller.data_mut().logs;
            if logs.len() < 512 {
                logs.push(LogLine {
                    level: level.as_str().to_string(),
                    message,
                });
            }
        },
    )?;

    linker.func_wrap(
        HOST_MODULE,
        "call",
        |mut caller: Caller<'_, HostState>,
         name_ptr: u32,
         name_len: u32,
         req_ptr: u32,
         req_len: u32|
         -> i32 {
            let max = caller.data().policy.limits.max_request_bytes;
            if req_len as usize > max || name_len > 256 {
                caller.data_mut().pending = respond::<()>(Err(CapError::invalid(format!(
                    "capability request of {req_len} bytes exceeds the {max} byte limit"
                ))));
                return caller.data().pending.len() as i32;
            }

            let Some(name_bytes) = read_guest(&mut caller, name_ptr, name_len) else {
                return error_code::BAD_POINTER;
            };
            let Some(req_bytes) = read_guest(&mut caller, req_ptr, req_len) else {
                return error_code::BAD_POINTER;
            };
            let Ok(name) = String::from_utf8(name_bytes) else {
                return error_code::BAD_REQUEST;
            };

            let response = dispatch(caller.data_mut(), &name, &req_bytes);
            caller.data_mut().pending = response;
            caller.data().pending.len() as i32
        },
    )?;

    linker.func_wrap(
        HOST_MODULE,
        "response_read",
        |mut caller: Caller<'_, HostState>, ptr: u32, len: u32| -> i32 {
            let pending = std::mem::take(&mut caller.data_mut().pending);
            if pending.len() != len as usize {
                caller.data_mut().pending = pending;
                return error_code::BAD_REQUEST;
            }
            if write_guest(&mut caller, ptr, &pending) {
                len as i32
            } else {
                error_code::BAD_POINTER
            }
        },
    )?;

    Ok(())
}

/// Route a capability call. Anything not listed here does not exist.
fn dispatch(state: &mut HostState, name: &str, request: &[u8]) -> Vec<u8> {
    match name {
        cap::TOOLS_LIST => respond(Ok(ToolsListResponse {
            tools: state.tools.specs(),
        })),

        cap::TOOL_INVOKE => {
            let parsed: Result<ToolInvokeRequest, _> = serde_json::from_slice(request);
            match parsed {
                Ok(req) => respond(
                    state
                        .tools
                        .invoke(&req.name, &req.input)
                        .map(|content| ToolInvokeResponse { content }),
                ),
                Err(e) => respond::<ToolInvokeResponse>(Err(CapError::invalid(format!(
                    "malformed tool.invoke request: {e}"
                )))),
            }
        }

        cap::LLM_COMPLETE => {
            // The guest counts its own steps, but the host does not take its word
            // for it: this is the ceiling that actually binds.
            let allowed = state.policy.limits.max_steps;
            if state.llm_calls >= allowed {
                return respond::<LlmResponse>(Err(CapError::denied(format!(
                    "this run has already used its {allowed} model turns"
                ))));
            }
            let parsed: Result<LlmRequest, _> = serde_json::from_slice(request);
            match parsed {
                Ok(req) => {
                    state.llm_calls += 1;
                    respond(state.llm.complete(&req))
                }
                Err(e) => respond::<LlmResponse>(Err(CapError::invalid(format!(
                    "malformed llm.complete request: {e}"
                )))),
            }
        }

        other => respond::<()>(Err(CapError::not_found(format!(
            "this host provides no capability named {other:?}"
        )))),
    }
}

fn respond<T: Serialize>(result: Result<T, CapError>) -> Vec<u8> {
    let envelope = match result {
        Ok(value) => CapResponse::Ok(value),
        Err(error) => CapResponse::Err(error),
    };
    serde_json::to_vec(&envelope).unwrap_or_else(|e| {
        let fallback = CapResponse::<()>::Err(CapError::new(
            "internal",
            format!("host could not encode its response: {e}"),
        ));
        serde_json::to_vec(&fallback).unwrap_or_else(|_| b"{\"err\":{}}".to_vec())
    })
}

// ---------------------------------------------------------------------------
// Guest memory access
// ---------------------------------------------------------------------------

fn guest_memory(caller: &mut Caller<'_, HostState>) -> Option<Memory> {
    caller.get_export("memory")?.into_memory()
}

/// Copy `len` bytes out of guest memory, refusing any range that is not wholly
/// inside it. Every pointer the guest hands us is treated as hostile.
fn read_guest(caller: &mut Caller<'_, HostState>, ptr: u32, len: u32) -> Option<Vec<u8>> {
    let memory = guest_memory(caller)?;
    let data = memory.data(&*caller);
    let start = ptr as usize;
    let end = start.checked_add(len as usize)?;
    data.get(start..end).map(<[u8]>::to_vec)
}

fn write_guest(caller: &mut Caller<'_, HostState>, ptr: u32, bytes: &[u8]) -> bool {
    let Some(memory) = guest_memory(caller) else {
        return false;
    };
    let data = memory.data_mut(&mut *caller);
    let start = ptr as usize;
    let Some(end) = start.checked_add(bytes.len()) else {
        return false;
    };
    match data.get_mut(start..end) {
        Some(slice) => {
            slice.copy_from_slice(bytes);
            true
        }
        None => false,
    }
}
