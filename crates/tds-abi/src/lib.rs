//! Shared ABI between the WebAssembly guest agent and the host runtime.
//!
//! The guest has no ambient authority: it cannot open sockets, read files, or
//! read the clock. Everything it needs arrives through [`host_call`]-style
//! capability requests that the host authorizes against a policy.
//!
//! # Wire protocol
//!
//! The guest exports:
//!
//! | export        | signature                  | purpose                                   |
//! |---------------|----------------------------|-------------------------------------------|
//! | `tds_abi_version` | `() -> i32`            | ABI handshake, must equal [`ABI_VERSION`]  |
//! | `tds_alloc`   | `(i32) -> i32`             | allocate `len` bytes in guest memory       |
//! | `tds_dealloc` | `(i32, i32)`               | free a previous allocation                 |
//! | `tds_run`     | `(i32, i32) -> i64`        | run the agent, returns packed `ptr`/`len`  |
//!
//! The host provides the `tds_host` module:
//!
//! | import               | signature                          | purpose                                    |
//! |----------------------|------------------------------------|--------------------------------------------|
//! | `log`                | `(i32, i32, i32)`                  | structured log line at a severity level     |
//! | `call`               | `(i32, i32, i32, i32) -> i32`      | invoke a capability, returns response bytes |
//! | `response_read`      | `(i32, i32) -> i32`                | copy the pending response into guest memory |
//!
//! `call` returns the byte length of the pending response, or a negative
//! [`ErrorCode`] if the host could not produce one at all. Capability-level
//! failures are *not* negative returns: they come back as a JSON
//! [`CapResponse::Err`] so the agent can reason about them and recover.

use serde::{Deserialize, Serialize};

/// Version of this ABI. The host refuses to run a guest that disagrees.
pub const ABI_VERSION: i32 = 1;

/// Name of the import module the host exposes to the guest.
pub const HOST_MODULE: &str = "tds_host";

/// Negative return values from `tds_host::call`.
pub mod error_code {
    /// The guest passed a pointer/length pair outside its own memory.
    pub const BAD_POINTER: i32 = -1;
    /// The capability name or request payload was not valid UTF-8 / JSON.
    pub const BAD_REQUEST: i32 = -2;
    /// The host could not serialize a response.
    pub const INTERNAL: i32 = -3;
}

/// Severity levels for `tds_host::log`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl LogLevel {
    pub fn from_i32(v: i32) -> LogLevel {
        match v {
            0 => LogLevel::Debug,
            2 => LogLevel::Warn,
            3 => LogLevel::Error,
            _ => LogLevel::Info,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// Pack a guest pointer and length into the single `i64` that `tds_run` returns.
pub fn pack_ptr_len(ptr: u32, len: u32) -> u64 {
    ((ptr as u64) << 32) | (len as u64)
}

/// Inverse of [`pack_ptr_len`].
pub fn unpack_ptr_len(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32)
}

// ---------------------------------------------------------------------------
// Capability envelope
// ---------------------------------------------------------------------------

/// Canonical capability names understood by the host.
pub mod cap {
    /// List the tool specifications the policy currently permits.
    pub const TOOLS_LIST: &str = "tools.list";
    /// Invoke one of those tools.
    pub const TOOL_INVOKE: &str = "tool.invoke";
    /// Ask the model for the next step. Credentials never cross into the guest.
    pub const LLM_COMPLETE: &str = "llm.complete";
}

/// Every capability call resolves to exactly one of these.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapResponse<T> {
    Ok(T),
    Err(CapError),
}

impl<T> CapResponse<T> {
    pub fn into_result(self) -> Result<T, CapError> {
        match self {
            CapResponse::Ok(v) => Ok(v),
            CapResponse::Err(e) => Err(e),
        }
    }
}

/// A capability failure the agent is expected to handle, not crash on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapError {
    /// Stable machine-readable kind, e.g. `denied`, `not_found`, `upstream`.
    pub kind: String,
    pub message: String,
}

impl CapError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        CapError {
            kind: kind.into(),
            message: message.into(),
        }
    }

    pub fn denied(message: impl Into<String>) -> Self {
        CapError::new("denied", message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        CapError::new("not_found", message)
    }

    pub fn upstream(message: impl Into<String>) -> Self {
        CapError::new("upstream", message)
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        CapError::new("invalid", message)
    }
}

impl std::fmt::Display for CapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

// ---------------------------------------------------------------------------
// Agent invocation
// ---------------------------------------------------------------------------

/// What the host hands the agent when it starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInput {
    /// The task the agent should accomplish.
    pub task: String,
    /// Optional extra context prepended to the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Hard ceiling on reason/act iterations, enforced by the guest and
    /// independently by the host's fuel budget.
    pub max_steps: u32,
}

/// What the agent hands back when it finishes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// Final natural-language answer.
    pub answer: String,
    /// How many model turns were consumed.
    pub steps: u32,
    /// Why the loop ended.
    pub stop: StopReason,
    /// Every tool the agent invoked, in order.
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    /// Token usage summed across all model turns.
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model produced a final answer.
    Completed,
    /// The agent hit its own `max_steps` ceiling.
    StepLimit,
    /// The model declined the request.
    Refusal,
    /// A capability failed in a way the agent could not work around.
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub step: u32,
    pub name: String,
    pub input: serde_json::Value,
    pub ok: bool,
    /// Truncated result, for the audit trail.
    pub result: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// A tool the host is willing to run on the agent's behalf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments.
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsListResponse {
    pub tools: Vec<ToolSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvokeRequest {
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvokeResponse {
    /// Text handed back to the model as a `tool_result` block.
    pub content: String,
}

// ---------------------------------------------------------------------------
// Model conversation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

/// One content block. Mirrors the Messages API block shapes so blocks the host
/// receives can be echoed back verbatim on the next turn — which matters for
/// `thinking` blocks, whose signatures must survive the round trip unchanged.
///
/// The untagged `Raw` arm means a block type this crate does not know about is
/// still carried through losslessly instead of failing to deserialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Known(KnownContent),
    Raw(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KnownContent {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

impl Content {
    pub fn text(text: impl Into<String>) -> Content {
        Content::Known(KnownContent::Text { text: text.into() })
    }

    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Content {
        Content::Known(KnownContent::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error,
        })
    }

    /// The text of this block, if it is a text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Known(KnownContent::Text { text }) => Some(text),
            _ => None,
        }
    }

    /// The `(id, name, input)` of this block, if it is a tool-use block.
    pub fn as_tool_use(&self) -> Option<(&str, &str, &serde_json::Value)> {
        match self {
            Content::Known(KnownContent::ToolUse { id, name, input }) => Some((id, name, input)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn user(content: Vec<Content>) -> Message {
        Message {
            role: Role::User,
            content,
        }
    }

    pub fn assistant(content: Vec<Content>) -> Message {
        Message {
            role: Role::Assistant,
            content,
        }
    }
}

/// A model turn requested by the agent. Note what is *absent*: no model id, no
/// API key, no endpoint. The host owns all three.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// `end_turn`, `tool_use`, `max_tokens`, `refusal`, ...
    pub stop_reason: String,
    pub content: Vec<Content>,
    #[serde(default)]
    pub usage: Usage,
}

impl LlmResponse {
    /// Concatenate every text block, which is what a caller wants as the answer.
    pub fn joined_text(&self) -> String {
        let parts: Vec<&str> = self.content.iter().filter_map(Content::as_text).collect();
        parts.join("\n").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptr_len_roundtrips() {
        let (p, l) = unpack_ptr_len(pack_ptr_len(0xdead_beef, 4096));
        assert_eq!((p, l), (0xdead_beef, 4096));
    }

    #[test]
    fn known_blocks_use_the_messages_api_shape() {
        let block = Content::Known(KnownContent::ToolUse {
            id: "toolu_1".into(),
            name: "http_get".into(),
            input: serde_json::json!({ "url": "https://example.com" }),
        });
        let wire = serde_json::to_value(&block).unwrap();
        assert_eq!(wire["type"], "tool_use");
        assert_eq!(wire["name"], "http_get");
    }

    #[test]
    fn unknown_blocks_survive_a_round_trip() {
        // A block type this crate has never heard of must still come back out
        // byte-identical, so future API additions cannot break the replay.
        let wire = serde_json::json!({ "type": "some_future_block", "payload": [1, 2, 3] });
        let parsed: Content = serde_json::from_value(wire.clone()).unwrap();
        assert!(matches!(parsed, Content::Raw(_)));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
    }

    #[test]
    fn thinking_signatures_survive_a_round_trip() {
        let wire = serde_json::json!({
            "type": "thinking",
            "thinking": "",
            "signature": "sig-abc"
        });
        let parsed: Content = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
    }
}
