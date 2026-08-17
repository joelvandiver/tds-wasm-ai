//! Integration tests for the sandbox boundary.
//!
//! These run the real agent module under the real runtime, with a scripted
//! model so the assertions are about the sandbox rather than about the weather
//! on the far side of an API call.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tds_abi::{
    AgentInput, CapError, Content, KnownContent, LlmRequest, LlmResponse, StopReason, Usage,
};
use tds_host::llm::LlmProvider;
use tds_host::policy::{Limits, Policy, Provider};
use tds_host::runtime::Runtime;

/// The compiled agent. Build it with `make agent` before running these tests.
fn agent_wasm() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tds-agent/target/wasm32-unknown-unknown/release/tds_agent.wasm");
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "could not read the agent module at {}: {e}\n\
             Build it first: make agent",
            path.display()
        )
    })
}

/// A model that replays a fixed script, so every test is deterministic.
struct ScriptedModel {
    turns: Vec<LlmResponse>,
    calls: Arc<AtomicUsize>,
}

impl ScriptedModel {
    fn new(turns: Vec<LlmResponse>) -> (Arc<ScriptedModel>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(ScriptedModel {
                turns,
                calls: calls.clone(),
            }),
            calls,
        )
    }
}

impl LlmProvider for ScriptedModel {
    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, CapError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        // Past the end of the script, keep replaying the last turn so a runaway
        // loop is bounded by the policy rather than by the script running dry.
        let turn = self.turns.get(n).or_else(|| self.turns.last());
        turn.cloned()
            .ok_or_else(|| CapError::upstream("script exhausted"))
    }

    fn describe(&self) -> String {
        "scripted".to_string()
    }
}

fn tool_use_turn(id: &str, name: &str, input: serde_json::Value) -> LlmResponse {
    LlmResponse {
        stop_reason: "tool_use".into(),
        content: vec![Content::Known(KnownContent::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        })],
        usage: Usage {
            input_tokens: 10,
            output_tokens: 5,
        },
    }
}

fn text_turn(text: &str) -> LlmResponse {
    LlmResponse {
        stop_reason: "end_turn".into(),
        content: vec![Content::text(text)],
        usage: Usage {
            input_tokens: 10,
            output_tokens: 5,
        },
    }
}

fn policy_with(enabled: &[&str], limits: Limits) -> Policy {
    let mut policy = Policy::default();
    policy.llm.provider = Provider::Mock;
    policy.limits = limits;
    policy.tools.enabled = enabled.iter().map(|s| s.to_string()).collect();
    policy
}

fn input(task: &str, max_steps: u32) -> AgentInput {
    AgentInput {
        task: task.into(),
        context: None,
        max_steps,
    }
}

#[test]
fn a_run_completes_and_records_every_tool_call() {
    let (model, calls) = ScriptedModel::new(vec![
        tool_use_turn(
            "t1",
            "kv_put",
            serde_json::json!({ "key": "k", "value": "v" }),
        ),
        text_turn("Saved it."),
    ]);
    let runtime = Runtime::new(
        policy_with(&["kv_put", "kv_get"], Limits::default()),
        &agent_wasm(),
        Box::new(ScriptedModelHandle(model)),
    )
    .unwrap();

    let report = runtime.run(input("save something", 4)).unwrap();

    assert_eq!(report.output.stop, StopReason::Completed);
    assert_eq!(report.output.answer, "Saved it.");
    assert_eq!(report.output.steps, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    assert_eq!(report.output.tool_calls.len(), 1);
    let call = &report.output.tool_calls[0];
    assert_eq!(call.name, "kv_put");
    assert!(call.ok, "kv_put should have succeeded: {}", call.result);

    // Usage is summed across turns, not just taken from the last one.
    assert_eq!(report.output.usage.input_tokens, 20);
    assert!(report.fuel_used > 0);
}

#[test]
fn a_denied_tool_reaches_the_agent_as_an_error_it_can_read() {
    // `http_get` is enabled, but the allowlist is empty, so the host refuses the
    // call. The run should continue and the denial should be legible.
    let (model, _) = ScriptedModel::new(vec![
        tool_use_turn(
            "t1",
            "http_get",
            serde_json::json!({ "url": "https://example.com" }),
        ),
        text_turn("I could not reach that host."),
    ]);
    let runtime = Runtime::new(
        policy_with(&["http_get"], Limits::default()),
        &agent_wasm(),
        Box::new(ScriptedModelHandle(model)),
    )
    .unwrap();

    let report = runtime.run(input("fetch a page", 4)).unwrap();

    assert_eq!(report.output.stop, StopReason::Completed);
    let call = &report.output.tool_calls[0];
    assert!(!call.ok);
    assert!(
        call.result.contains("denied") && call.result.contains("allowlist"),
        "denial should say what was refused and why: {}",
        call.result
    );
}

#[test]
fn the_host_step_ceiling_binds_even_when_the_caller_asks_for_more() {
    // The model never stops asking for tools. The policy allows two turns; the
    // caller requests a hundred.
    let (model, calls) =
        ScriptedModel::new(vec![tool_use_turn("t1", "now", serde_json::json!({}))]);
    let limits = Limits {
        max_steps: 2,
        ..Default::default()
    };
    let runtime = Runtime::new(
        policy_with(&["now"], limits),
        &agent_wasm(),
        Box::new(ScriptedModelHandle(model)),
    )
    .unwrap();

    let report = runtime.run(input("loop forever", 100)).unwrap();

    assert_eq!(report.output.stop, StopReason::StepLimit);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the policy ceiling should bind, not the request"
    );
}

#[test]
fn a_module_that_imports_wasi_is_refused_before_it_runs() {
    let wasm = wat::parse_str(
        r#"
        (module
          (import "wasi_snapshot_preview1" "fd_write"
            (func $fd_write (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "tds_abi_version") (result i32) (i32.const 1))
          (func (export "tds_alloc") (param i32) (result i32) (i32.const 1024))
          (func (export "tds_dealloc") (param i32 i32))
          (func (export "tds_run") (param i32 i32) (result i64) (i64.const 0)))
        "#,
    )
    .unwrap();

    let (model, _) = ScriptedModel::new(vec![text_turn("unused")]);
    let error = expect_rejected(
        Runtime::new(
            policy_with(&[], Limits::default()),
            &wasm,
            Box::new(ScriptedModelHandle(model)),
        ),
        "a module importing WASI must be rejected",
    );

    let message = format!("{error:#}");
    assert!(
        message.contains("wasi_snapshot_preview1"),
        "the error should name the offending import: {message}"
    );
}

#[test]
fn a_module_missing_the_abi_exports_is_refused() {
    let wasm = wat::parse_str(r#"(module (memory (export "memory") 1))"#).unwrap();
    let (model, _) = ScriptedModel::new(vec![text_turn("unused")]);

    let error = expect_rejected(
        Runtime::new(
            policy_with(&[], Limits::default()),
            &wasm,
            Box::new(ScriptedModelHandle(model)),
        ),
        "a module without the ABI exports must be rejected",
    );

    let message = format!("{error:#}");
    assert!(
        message.contains("does not export"),
        "the error should name the missing export: {message}"
    );
}

#[test]
fn a_spinning_module_is_stopped_by_its_compute_budget() {
    // The guest never calls back into the host, so nothing but fuel can stop it.
    let wasm = wat::parse_str(
        r#"
        (module
          (memory (export "memory") 1)
          (func (export "tds_abi_version") (result i32) (i32.const 1))
          (func (export "tds_alloc") (param i32) (result i32) (i32.const 1024))
          (func (export "tds_dealloc") (param i32 i32))
          (func (export "tds_run") (param i32 i32) (result i64)
            (loop $spin (br $spin))
            (i64.const 0)))
        "#,
    )
    .unwrap();

    let (model, _) = ScriptedModel::new(vec![text_turn("unused")]);
    let limits = Limits {
        fuel: 5_000_000,
        ..Default::default()
    };
    let runtime = Runtime::new(
        policy_with(&[], limits),
        &wasm,
        Box::new(ScriptedModelHandle(model)),
    )
    .unwrap();

    let report = runtime.run(input("spin", 1)).unwrap();

    assert_eq!(report.output.stop, StopReason::Error);
    assert!(
        report.output.answer.contains("compute budget"),
        "the operator should be told which budget ran out: {}",
        report.output.answer
    );
}

#[test]
fn each_run_gets_a_fresh_scratchpad() {
    // A value written in one run must not be visible in the next.
    let make_runtime = |script: Vec<LlmResponse>| {
        let (model, _) = ScriptedModel::new(script);
        Runtime::new(
            policy_with(&["kv_put", "kv_get"], Limits::default()),
            &agent_wasm(),
            Box::new(ScriptedModelHandle(model)),
        )
        .unwrap()
    };

    let writer = make_runtime(vec![
        tool_use_turn(
            "t1",
            "kv_put",
            serde_json::json!({ "key": "secret", "value": "hunter2" }),
        ),
        text_turn("stored"),
    ]);
    writer.run(input("write", 4)).unwrap();

    let reader = make_runtime(vec![
        tool_use_turn("t1", "kv_get", serde_json::json!({ "key": "secret" })),
        text_turn("read"),
    ]);
    let report = reader.run(input("read", 4)).unwrap();

    let result = &report.output.tool_calls[0].result;
    assert!(
        !result.contains("hunter2"),
        "state must not leak between runs, got: {result}"
    );
}

/// `Runtime` is not `Debug`, so unwrap the rejection by hand.
fn expect_rejected(result: anyhow::Result<Runtime>, why: &str) -> anyhow::Error {
    match result {
        Ok(_) => panic!("{why}"),
        Err(e) => e,
    }
}

/// `Runtime::new` takes an owned provider; this hands it our shared script.
struct ScriptedModelHandle(Arc<ScriptedModel>);

impl LlmProvider for ScriptedModelHandle {
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, CapError> {
        self.0.complete(request)
    }
    fn describe(&self) -> String {
        self.0.describe()
    }
}
