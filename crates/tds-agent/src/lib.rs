//! The encapsulated AI agent.
//!
//! This crate compiles to a `wasm32-unknown-unknown` module with no WASI and no
//! syscalls of any kind. Its only imports are the three `tds_host` functions in
//! [`tds_abi`]. It cannot open a socket, read a file, or learn the time except
//! by asking the host, which answers only what its policy allows.
//!
//! The agent itself is an ordinary reason-and-act loop: ask the model what to do,
//! run any tools it asks for, feed the results back, repeat until it answers.

use std::alloc::Layout;

use serde::de::DeserializeOwned;
use serde::Serialize;
use tds_abi::{
    cap, error_code, pack_ptr_len, AgentInput, AgentOutput, CapError, CapResponse, Content,
    LlmRequest, LlmResponse, LogLevel, Message, StopReason, ToolCallRecord, ToolInvokeRequest,
    ToolInvokeResponse, ToolSpec, ToolsListResponse, Usage, ABI_VERSION,
};

#[link(wasm_import_module = "tds_host")]
extern "C" {
    fn log(level: i32, ptr: u32, len: u32);
    fn call(name_ptr: u32, name_len: u32, req_ptr: u32, req_len: u32) -> i32;
    fn response_read(ptr: u32, len: u32) -> i32;
}

// ---------------------------------------------------------------------------
// Guest exports
// ---------------------------------------------------------------------------

/// ABI handshake. The host refuses to run a module that answers differently.
#[no_mangle]
pub extern "C" fn tds_abi_version() -> i32 {
    ABI_VERSION
}

/// Allocate `len` bytes of guest memory for the host to write into.
#[no_mangle]
pub extern "C" fn tds_alloc(len: u32) -> u32 {
    if len == 0 {
        return 1; // non-null dangling; never dereferenced
    }
    match Layout::from_size_align(len as usize, 1) {
        Ok(layout) => unsafe { std::alloc::alloc(layout) as u32 },
        Err(_) => 0,
    }
}

/// Free an allocation previously handed out by [`tds_alloc`].
///
/// # Safety
/// `ptr`/`len` must come from a matching `tds_alloc` call that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tds_dealloc(ptr: u32, len: u32) {
    if ptr == 0 || ptr == 1 || len == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len as usize, 1) {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

/// Run the agent. Takes JSON-encoded [`AgentInput`], returns JSON-encoded
/// [`AgentOutput`] as a packed pointer/length the host unpacks and then frees.
///
/// # Safety
/// `ptr`/`len` must describe a readable region of guest memory.
#[no_mangle]
pub unsafe extern "C" fn tds_run(ptr: u32, len: u32) -> u64 {
    let raw = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    let output = match serde_json::from_slice::<AgentInput>(raw) {
        Ok(input) => run_agent(input),
        Err(e) => AgentOutput {
            answer: format!("could not parse agent input: {e}"),
            steps: 0,
            stop: StopReason::Error,
            tool_calls: Vec::new(),
            usage: Usage::default(),
        },
    };

    let bytes = serde_json::to_vec(&output).unwrap_or_else(|_| {
        br#"{"answer":"serialization failed","steps":0,"stop":"error"}"#.to_vec()
    });
    leak_to_host(bytes)
}

/// Hand a buffer to the host, which reads it and calls [`tds_dealloc`].
fn leak_to_host(bytes: Vec<u8>) -> u64 {
    let len = bytes.len() as u32;
    let out = tds_alloc(len);
    if out == 0 {
        return pack_ptr_len(0, 0);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, len as usize);
    }
    pack_ptr_len(out, len)
}

// ---------------------------------------------------------------------------
// Host capability helpers
// ---------------------------------------------------------------------------

fn emit(level: LogLevel, message: &str) {
    let bytes = message.as_bytes();
    unsafe { log(level as i32, bytes.as_ptr() as u32, bytes.len() as u32) };
}

/// Invoke a host capability and decode its response.
fn host_call<Req, Res>(name: &str, request: &Req) -> Result<Res, CapError>
where
    Req: Serialize,
    Res: DeserializeOwned,
{
    let body = serde_json::to_vec(request)
        .map_err(|e| CapError::invalid(format!("could not encode {name} request: {e}")))?;

    let n = unsafe {
        call(
            name.as_ptr() as u32,
            name.len() as u32,
            body.as_ptr() as u32,
            body.len() as u32,
        )
    };

    if n < 0 {
        let reason = match n {
            error_code::BAD_POINTER => "host rejected a memory reference",
            error_code::BAD_REQUEST => "host could not decode the request",
            error_code::INTERNAL => "host failed internally",
            _ => "unknown host failure",
        };
        return Err(CapError::new("host", format!("{name}: {reason}")));
    }

    let mut buf = vec![0u8; n as usize];
    if n > 0 {
        let copied = unsafe { response_read(buf.as_mut_ptr() as u32, n as u32) };
        if copied != n {
            return Err(CapError::new("host", format!("{name}: truncated response")));
        }
    }

    serde_json::from_slice::<CapResponse<Res>>(&buf)
        .map_err(|e| CapError::new("host", format!("{name}: malformed response: {e}")))?
        .into_result()
}

// ---------------------------------------------------------------------------
// The agent
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are an autonomous agent running inside a sandboxed WebAssembly module. You \
have no direct access to the network, the filesystem, or the clock — the only \
way to affect or observe anything outside your own memory is the tools listed \
below, each of which the host authorizes individually.

Work the task to completion. Call a tool when you need information you do not \
already have, and prefer one well-chosen call over several speculative ones. If \
a tool returns an error, read it: a denial means the host's policy forbids that \
action and retrying it will fail the same way, so find another route or explain \
what you would need.

When you have the answer, state it directly. Lead with the outcome, then any \
supporting detail. Do not narrate your process or describe the tools you used \
unless the task asked for that.";

fn run_agent(input: AgentInput) -> AgentOutput {
    let tools: Vec<ToolSpec> =
        match host_call::<_, ToolsListResponse>(cap::TOOLS_LIST, &serde_json::json!({})) {
            Ok(list) => list.tools,
            Err(e) => {
                emit(LogLevel::Warn, &format!("no tools available: {e}"));
                Vec::new()
            }
        };
    emit(
        LogLevel::Info,
        &format!("agent starting with {} tool(s)", tools.len()),
    );

    let mut messages = Vec::new();
    if let Some(context) = input.context.as_deref().filter(|c| !c.trim().is_empty()) {
        messages.push(Message::user(vec![Content::text(format!(
            "Background context:\n{context}"
        ))]));
    }
    messages.push(Message::user(vec![Content::text(input.task.clone())]));

    let mut record = Vec::new();
    let mut usage = Usage::default();
    let max_steps = input.max_steps.max(1);

    for step in 1..=max_steps {
        let request = LlmRequest {
            system: Some(SYSTEM_PROMPT.to_string()),
            messages: messages.clone(),
            tools: tools.clone(),
        };

        let response: LlmResponse = match host_call(cap::LLM_COMPLETE, &request) {
            Ok(r) => r,
            Err(e) => {
                emit(LogLevel::Error, &format!("model turn {step} failed: {e}"));
                return AgentOutput {
                    answer: format!("The model call failed at step {step}: {e}"),
                    steps: step - 1,
                    stop: StopReason::Error,
                    tool_calls: record,
                    usage,
                };
            }
        };
        usage.add(response.usage);

        if response.stop_reason == "refusal" {
            return AgentOutput {
                answer: response.joined_text(),
                steps: step,
                stop: StopReason::Refusal,
                tool_calls: record,
                usage,
            };
        }

        // Echo the assistant turn back verbatim. Thinking blocks in particular
        // must survive unedited or the next turn is rejected.
        messages.push(Message::assistant(response.content.clone()));

        let requested: Vec<(String, String, serde_json::Value)> = response
            .content
            .iter()
            .filter_map(Content::as_tool_use)
            .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
            .collect();

        if requested.is_empty() {
            return AgentOutput {
                answer: response.joined_text(),
                steps: step,
                stop: StopReason::Completed,
                tool_calls: record,
                usage,
            };
        }

        // Every tool_result for a turn goes back in a single user message.
        let mut results = Vec::with_capacity(requested.len());
        for (id, name, args) in requested {
            emit(LogLevel::Info, &format!("step {step}: calling tool {name}"));
            let invocation = ToolInvokeRequest {
                name: name.clone(),
                input: args.clone(),
            };
            let (text, ok) = match host_call::<_, ToolInvokeResponse>(cap::TOOL_INVOKE, &invocation)
            {
                Ok(r) => (r.content, true),
                // A denial or upstream failure is reported to the model as an
                // error result so it can adapt, rather than ending the run.
                Err(e) => (e.to_string(), false),
            };
            record.push(ToolCallRecord {
                step,
                name,
                input: args,
                ok,
                result: truncate(&text, 512),
            });
            results.push(Content::tool_result(id, text, !ok));
        }
        messages.push(Message::user(results));
    }

    emit(LogLevel::Warn, "agent hit its step limit");
    AgentOutput {
        answer: "Stopped after reaching the step limit without producing a final answer."
            .to_string(),
        steps: max_steps,
        stop: StopReason::StepLimit,
        tool_calls: record,
        usage,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [{} bytes truncated]", &s[..end], s.len() - end)
}
