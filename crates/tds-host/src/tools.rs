//! Tools the host is willing to run on the agent's behalf.
//!
//! Each tool is a deliberate hole in the sandbox wall, so each one is narrow:
//! the agent names a tool and passes arguments, and the host decides whether
//! that particular call is permitted before anything happens.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

use tds_abi::{CapError, ToolSpec};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::policy::Policy;

/// Every tool this host knows how to run. A policy may enable any subset.
pub const KNOWN_TOOLS: &[&str] = &["now", "http_get", "kv_get", "kv_put"];

pub fn is_known_tool(name: &str) -> bool {
    KNOWN_TOOLS.contains(&name)
}

/// Per-run tool state. A fresh box is built for every agent run, so the key/value
/// scratchpad never leaks between runs.
pub struct ToolBox {
    policy: Arc<Policy>,
    http: reqwest::blocking::Client,
    kv: BTreeMap<String, String>,
}

impl ToolBox {
    pub fn new(policy: Arc<Policy>) -> Result<ToolBox, anyhow::Error> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(policy.tools.http.timeout_ms))
            .user_agent(concat!("tds-wasm-ai/", env!("CARGO_PKG_VERSION")))
            // Redirects are not followed: the destination of a redirect has not
            // been checked against the allowlist. The agent gets the Location
            // back and can fetch it explicitly, which re-runs the check.
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(ToolBox {
            policy,
            http,
            kv: BTreeMap::new(),
        })
    }

    /// The tool surface the agent is shown. A disabled tool is not described,
    /// not listed, and does not exist as far as the model is concerned.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.policy
            .tools
            .enabled
            .iter()
            .filter_map(|name| spec_for(name))
            .collect()
    }

    pub fn invoke(&mut self, name: &str, input: &serde_json::Value) -> Result<String, CapError> {
        if !self.policy.tools.enabled.iter().any(|t| t == name) {
            return Err(CapError::denied(format!(
                "tool {name:?} is not enabled by this host's policy"
            )));
        }
        match name {
            "now" => Ok(now_rfc3339()),
            "http_get" => self.http_get(input),
            "kv_get" => self.kv_get(input),
            "kv_put" => self.kv_put(input),
            other => Err(CapError::not_found(format!("no tool named {other:?}"))),
        }
    }

    fn http_get(&self, input: &serde_json::Value) -> Result<String, CapError> {
        let raw = string_arg(input, "url")?;
        let parsed = url::Url::parse(&raw)
            .map_err(|e| CapError::invalid(format!("url is not valid: {e}")))?;

        if parsed.scheme() != "https" {
            return Err(CapError::denied("only https URLs may be fetched"));
        }
        let host = parsed
            .host_str()
            .ok_or_else(|| CapError::invalid("url has no host"))?;
        if !self.policy.tools.http.allows(host) {
            return Err(CapError::denied(format!(
                "host {host:?} is not in this host's allowlist"
            )));
        }

        let response = self
            .http
            .get(parsed.clone())
            .send()
            .map_err(|e| CapError::upstream(format!("GET {host} failed: {e}")))?;

        let status = response.status();
        if status.is_redirection() {
            let location = response
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("(none)");
            return Ok(format!(
                "HTTP {} redirect to {location}. Redirects are not followed automatically; \
                 fetch that URL explicitly if you still want it.",
                status.as_u16()
            ));
        }

        let cap = self.policy.tools.http.max_response_bytes;
        let mut body = Vec::with_capacity(cap.min(16 * 1024));
        response
            .take(cap as u64)
            .read_to_end(&mut body)
            .map_err(|e| CapError::upstream(format!("reading response body: {e}")))?;

        let truncated = body.len() >= cap;
        let text = String::from_utf8_lossy(&body);
        Ok(format!(
            "HTTP {}{}\n\n{}",
            status.as_u16(),
            if truncated {
                format!(" (truncated to {cap} bytes)")
            } else {
                String::new()
            },
            text
        ))
    }

    fn kv_get(&self, input: &serde_json::Value) -> Result<String, CapError> {
        let key = string_arg(input, "key")?;
        Ok(self
            .kv
            .get(&key)
            .cloned()
            .unwrap_or_else(|| format!("(no value stored for {key:?})")))
    }

    fn kv_put(&mut self, input: &serde_json::Value) -> Result<String, CapError> {
        let key = string_arg(input, "key")?;
        let value = string_arg(input, "value")?;
        let limits = &self.policy.tools.kv;

        if value.len() > limits.max_value_bytes {
            return Err(CapError::denied(format!(
                "value is {} bytes, over the {} byte limit",
                value.len(),
                limits.max_value_bytes
            )));
        }
        if !self.kv.contains_key(&key) && self.kv.len() >= limits.max_keys {
            return Err(CapError::denied(format!(
                "the scratchpad already holds its maximum of {} keys",
                limits.max_keys
            )));
        }
        self.kv.insert(key.clone(), value);
        Ok(format!("stored {key:?}"))
    }
}

fn string_arg(input: &serde_json::Value, field: &str) -> Result<String, CapError> {
    input
        .get(field)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| CapError::invalid(format!("missing required string argument {field:?}")))
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unavailable".to_string())
}

fn spec_for(name: &str) -> Option<ToolSpec> {
    let (description, schema) = match name {
        "now" => (
            "Get the current UTC time as an RFC 3339 timestamp. The sandbox has no \
             clock of its own, so this is the only way to learn the time.",
            serde_json::json!({ "type": "object", "properties": {}, "required": [] }),
        ),
        "http_get" => (
            "Fetch a URL over HTTPS and return the status and body. Only hosts on the \
             runtime's allowlist can be reached; anything else is denied. Responses are \
             truncated past a size limit and redirects are reported rather than followed.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Absolute https:// URL to fetch." }
                },
                "required": ["url"]
            }),
        ),
        "kv_get" => (
            "Read a value previously saved with kv_put. Returns a placeholder if the key \
             has never been set. The scratchpad lasts only for this run.",
            serde_json::json!({
                "type": "object",
                "properties": { "key": { "type": "string" } },
                "required": ["key"]
            }),
        ),
        "kv_put" => (
            "Save a string under a key so you can retrieve it later in this run. Use it to \
             park intermediate findings instead of restating them each turn.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string" },
                    "value": { "type": "string" }
                },
                "required": ["key", "value"]
            }),
        ),
        _ => return None,
    };

    Some(ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: schema,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{HttpToolConfig, ToolsConfig};

    fn toolbox(enabled: &[&str], allowed_hosts: &[&str]) -> ToolBox {
        let policy = Policy {
            tools: ToolsConfig {
                enabled: enabled.iter().map(|s| s.to_string()).collect(),
                http: HttpToolConfig {
                    allowed_hosts: allowed_hosts.iter().map(|s| s.to_string()).collect(),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        ToolBox::new(Arc::new(policy)).unwrap()
    }

    #[test]
    fn a_disabled_tool_is_denied_even_though_the_host_implements_it() {
        let mut tools = toolbox(&["now"], &[]);
        let err = tools.invoke(
            "http_get",
            &serde_json::json!({ "url": "https://example.com" }),
        );
        assert_eq!(err.unwrap_err().kind, "denied");
    }

    #[test]
    fn a_disabled_tool_is_not_described_to_the_model() {
        let tools = toolbox(&["now", "kv_get"], &[]);
        let names: Vec<String> = tools.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["now", "kv_get"]);
    }

    #[test]
    fn non_https_and_off_allowlist_urls_are_refused_before_any_request() {
        let mut tools = toolbox(&["http_get"], &["example.com"]);

        let plaintext = tools
            .invoke(
                "http_get",
                &serde_json::json!({ "url": "http://example.com" }),
            )
            .unwrap_err();
        assert_eq!(plaintext.kind, "denied");

        let elsewhere = tools
            .invoke(
                "http_get",
                &serde_json::json!({ "url": "https://evil.test/x" }),
            )
            .unwrap_err();
        assert_eq!(elsewhere.kind, "denied");

        // Loopback is not special-cased; it simply is not on the allowlist.
        let loopback = tools
            .invoke(
                "http_get",
                &serde_json::json!({ "url": "https://127.0.0.1/" }),
            )
            .unwrap_err();
        assert_eq!(loopback.kind, "denied");
    }

    #[test]
    fn the_scratchpad_round_trips_and_enforces_its_caps() {
        let mut tools = toolbox(&["kv_get", "kv_put"], &[]);
        tools
            .invoke(
                "kv_put",
                &serde_json::json!({ "key": "a", "value": "hello" }),
            )
            .unwrap();
        assert_eq!(
            tools
                .invoke("kv_get", &serde_json::json!({ "key": "a" }))
                .unwrap(),
            "hello"
        );

        let oversized = "x".repeat(32 * 1024);
        let err = tools
            .invoke(
                "kv_put",
                &serde_json::json!({ "key": "b", "value": oversized }),
            )
            .unwrap_err();
        assert_eq!(err.kind, "denied");
    }

    #[test]
    fn missing_arguments_are_reported_as_invalid_not_as_a_panic() {
        let mut tools = toolbox(&["kv_get"], &[]);
        let err = tools.invoke("kv_get", &serde_json::json!({})).unwrap_err();
        assert_eq!(err.kind, "invalid");
    }
}
