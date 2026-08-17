//! Model access, held entirely on the host side of the sandbox.
//!
//! The guest asks for a completion; the host decides which model answers, with
//! which credentials, at which endpoint. The API key is read from the host
//! environment and never enters guest memory.

use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use tds_abi::{CapError, Content, LlmRequest, LlmResponse, Usage};

use crate::policy::{LlmConfig, Provider, Thinking};

/// Beta flag for the server-side `fallbacks` parameter. On a policy decline the
/// API re-runs the request on a recommended model inside the same call, so a
/// benign request that trips a classifier still gets answered.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

pub trait LlmProvider: Send + Sync {
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, CapError>;
    fn describe(&self) -> String;
}

pub fn build(config: &LlmConfig) -> Result<Box<dyn LlmProvider>> {
    match config.provider {
        Provider::Mock => Ok(Box::new(MockProvider)),
        Provider::Anthropic => Ok(Box::new(AnthropicProvider::new(config.clone())?)),
    }
}

// ---------------------------------------------------------------------------
// Anthropic Messages API
// ---------------------------------------------------------------------------

pub struct AnthropicProvider {
    config: LlmConfig,
    api_key: String,
    client: reqwest::blocking::Client,
}

impl AnthropicProvider {
    pub fn new(config: LlmConfig) -> Result<AnthropicProvider> {
        let api_key = std::env::var(&config.api_key_env).with_context(|| {
            format!(
                "{} is not set; export an API key or set llm.provider = \"mock\" in the policy",
                config.api_key_env
            )
        })?;
        if api_key.trim().is_empty() {
            anyhow::bail!("{} is set but empty", config.api_key_env);
        }

        // The per-request timeout is generous on purpose: a single agentic turn
        // at high effort can legitimately run for minutes.
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .user_agent(concat!("tds-wasm-ai/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("building HTTP client")?;

        Ok(AnthropicProvider {
            config,
            api_key,
            client,
        })
    }

    fn body(&self, request: &LlmRequest) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": request.messages,
            "output_config": { "effort": self.config.effort.as_str() },
        });

        // Note what is not here: no temperature, top_p, or top_k. Current models
        // reject them outright.
        match self.config.thinking {
            Thinking::Adaptive => {
                let mut thinking = serde_json::json!({ "type": "adaptive" });
                if self.config.summarize_thinking {
                    thinking["display"] = serde_json::json!("summarized");
                }
                body["thinking"] = thinking;
            }
            Thinking::Disabled => {
                body["thinking"] = serde_json::json!({ "type": "disabled" });
            }
        }

        if let Some(system) = &request.system {
            body["system"] = serde_json::json!(system);
        }
        if !request.tools.is_empty() {
            body["tools"] = serde_json::json!(request.tools);
        }
        if self.config.server_side_fallback {
            body["fallbacks"] = serde_json::json!("default");
        }
        body
    }
}

impl LlmProvider for AnthropicProvider {
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, CapError> {
        let url = format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'));
        let body = self.body(request);

        let mut attempt = 0u32;
        loop {
            let mut http = self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", &self.config.api_version)
                .header("content-type", "application/json");
            if self.config.server_side_fallback {
                http = http.header("anthropic-beta", FALLBACK_BETA);
            }

            let response = match http.json(&body).send() {
                Ok(r) => r,
                Err(e) => {
                    if attempt < self.config.max_retries {
                        attempt += 1;
                        std::thread::sleep(backoff(attempt));
                        continue;
                    }
                    return Err(CapError::upstream(format!(
                        "request to the model failed: {e}"
                    )));
                }
            };

            let status = response.status();
            if status.is_success() {
                let parsed: ApiResponse = response
                    .json()
                    .map_err(|e| CapError::upstream(format!("malformed model response: {e}")))?;
                return Ok(parsed.into());
            }

            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok());
            let detail = response.text().unwrap_or_default();

            // 429 and 5xx are transient; 4xx means the request itself is wrong
            // and resending it unchanged will fail identically.
            let retryable = status.as_u16() == 429 || status.is_server_error();
            if retryable && attempt < self.config.max_retries {
                attempt += 1;
                let wait = retry_after
                    .map(Duration::from_secs)
                    .unwrap_or_else(|| backoff(attempt));
                tracing::warn!(status = status.as_u16(), attempt, "retrying model request");
                std::thread::sleep(wait);
                continue;
            }

            return Err(CapError::upstream(format!(
                "model returned HTTP {}: {}",
                status.as_u16(),
                truncate(&detail, 512)
            )));
        }
    }

    fn describe(&self) -> String {
        format!("anthropic:{}", self.config.model)
    }
}

fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(500u64.saturating_mul(1 << attempt.min(5)))
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    content: Vec<Content>,
    #[serde(default)]
    usage: ApiUsage,
}

#[derive(Debug, Default, Deserialize)]
struct ApiUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

impl From<ApiResponse> for LlmResponse {
    fn from(r: ApiResponse) -> LlmResponse {
        LlmResponse {
            stop_reason: r.stop_reason.unwrap_or_else(|| "end_turn".to_string()),
            content: r.content,
            usage: Usage {
                input_tokens: r.usage.input_tokens,
                output_tokens: r.usage.output_tokens,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Mock provider
// ---------------------------------------------------------------------------

/// A deterministic stand-in so the sandbox, the tool surface, and the agent loop
/// can be exercised end to end with no network and no API key.
///
/// It drives the loop from directives embedded in the task: a line of the form
/// `!tool <name> <json-args>` becomes a tool call on the first turn. Once the
/// results come back, it answers with a summary.
pub struct MockProvider;

impl LlmProvider for MockProvider {
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, CapError> {
        let already_used_tools = request.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::Known(tds_abi::KnownContent::ToolResult { .. })))
        });

        let task = request
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(Content::as_text)
            .collect::<Vec<_>>()
            .join("\n");

        if !already_used_tools {
            let directives = parse_directives(&task);
            let available: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
            let calls: Vec<Content> = directives
                .into_iter()
                .filter(|(name, _)| available.contains(&name.as_str()))
                .enumerate()
                .map(|(i, (name, input))| {
                    Content::Known(tds_abi::KnownContent::ToolUse {
                        id: format!("toolu_mock_{i}"),
                        name,
                        input,
                    })
                })
                .collect();

            if !calls.is_empty() {
                return Ok(LlmResponse {
                    stop_reason: "tool_use".to_string(),
                    content: calls,
                    usage: Usage {
                        input_tokens: 100,
                        output_tokens: 20,
                    },
                });
            }
        }

        let results: Vec<String> = request
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| match c {
                Content::Known(tds_abi::KnownContent::ToolResult { content, .. }) => {
                    Some(content.clone())
                }
                _ => None,
            })
            .collect();

        let answer = if results.is_empty() {
            format!(
                "[mock] No tools were needed. Task was: {}",
                first_line(&task)
            )
        } else {
            format!("[mock] Tool results: {}", results.join(" | "))
        };

        Ok(LlmResponse {
            stop_reason: "end_turn".to_string(),
            content: vec![Content::text(answer)],
            usage: Usage {
                input_tokens: 120,
                output_tokens: 30,
            },
        })
    }

    fn describe(&self) -> String {
        "mock".to_string()
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("").trim()
}

/// Pull `!tool <name> <json>` directives out of the task text.
fn parse_directives(task: &str) -> Vec<(String, serde_json::Value)> {
    task.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("!tool ")?;
            let (name, args) = match rest.split_once(char::is_whitespace) {
                Some((name, args)) => (name.trim(), args.trim()),
                None => (rest.trim(), "{}"),
            };
            let input = serde_json::from_str(args).unwrap_or(serde_json::json!({}));
            Some((name.to_string(), input))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tds_abi::{Message, ToolSpec};

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "test".into(),
            input_schema: serde_json::json!({ "type": "object" }),
        }
    }

    #[test]
    fn directives_parse_with_and_without_arguments() {
        let parsed = parse_directives("do a thing\n!tool now\n!tool kv_put {\"key\":\"a\"}\n");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, "now");
        assert_eq!(parsed[1].1["key"], "a");
    }

    #[test]
    fn the_mock_only_calls_tools_the_policy_exposed() {
        let request = LlmRequest {
            system: None,
            messages: vec![Message::user(vec![Content::text(
                "!tool now\n!tool http_get",
            )])],
            tools: vec![spec("now")],
        };
        let response = MockProvider.complete(&request).unwrap();
        assert_eq!(response.stop_reason, "tool_use");
        // `http_get` was requested but not offered, so it is dropped.
        assert_eq!(response.content.len(), 1);
        assert_eq!(response.content[0].as_tool_use().unwrap().1, "now");
    }

    #[test]
    fn the_mock_answers_once_results_come_back() {
        let request = LlmRequest {
            system: None,
            messages: vec![
                Message::user(vec![Content::text("!tool now")]),
                Message::user(vec![Content::tool_result(
                    "toolu_mock_0",
                    "2026-01-01",
                    false,
                )]),
            ],
            tools: vec![spec("now")],
        };
        let response = MockProvider.complete(&request).unwrap();
        assert_eq!(response.stop_reason, "end_turn");
        assert!(response.joined_text().contains("2026-01-01"));
    }

    #[test]
    fn the_request_body_omits_sampling_parameters() {
        let provider = AnthropicProvider {
            config: LlmConfig::default(),
            api_key: "test".into(),
            client: reqwest::blocking::Client::new(),
        };
        let body = provider.body(&LlmRequest {
            system: Some("sys".into()),
            messages: vec![Message::user(vec![Content::text("hi")])],
            tools: vec![spec("now")],
        });
        assert_eq!(body["model"], "claude-opus-5");
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["fallbacks"], "default");
        for rejected in ["temperature", "top_p", "top_k", "budget_tokens"] {
            assert!(body.get(rejected).is_none(), "{rejected} must not be sent");
        }
    }
}
