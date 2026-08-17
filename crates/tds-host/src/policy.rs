//! The sandbox policy: everything the agent is allowed to consume and reach.
//!
//! The policy is data, not code. Nothing in the guest can read it, argue with
//! it, or change it — the host consults it on every capability call.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
}

impl Policy {
    pub fn load(path: &Path) -> Result<Policy> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading policy {}", path.display()))?;
        let policy: Policy =
            toml::from_str(&text).with_context(|| format!("parsing policy {}", path.display()))?;
        policy.validate()?;
        Ok(policy)
    }

    /// Reject configurations the API would reject at request time, so a bad
    /// policy fails at startup instead of halfway through someone's run.
    pub fn validate(&self) -> Result<()> {
        if self.limits.fuel == 0 {
            bail!("limits.fuel must be greater than zero");
        }
        if self.limits.wall_clock_ms == 0 {
            bail!("limits.wall_clock_ms must be greater than zero");
        }
        if self.limits.max_steps == 0 {
            bail!("limits.max_steps must be greater than zero");
        }
        if self.limits.memory_bytes < 1 << 20 {
            bail!("limits.memory_bytes must be at least 1 MiB");
        }

        // Claude Opus 5 rejects disabled thinking above `high` effort.
        if self.llm.thinking == Thinking::Disabled
            && matches!(self.llm.effort, Effort::XHigh | Effort::Max)
        {
            bail!(
                "llm.thinking = \"disabled\" is only valid at effort \"high\" or lower \
                 (got {:?}); raise thinking or lower effort",
                self.llm.effort
            );
        }

        for name in &self.tools.enabled {
            if !crate::tools::is_known_tool(name) {
                bail!("tools.enabled lists unknown tool {name:?}");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// Wasmtime fuel budget for the whole run. Bounds guest compute.
    pub fuel: u64,
    /// Total wall-clock budget for the run, enforced by epoch interruption.
    pub wall_clock_ms: u64,
    /// Ceiling on the guest's linear memory.
    pub memory_bytes: usize,
    /// Ceiling on guest table elements (indirect-call slots).
    pub table_elements: usize,
    /// Host-side cap on model turns, independent of what the guest asks for.
    pub max_steps: u32,
    /// Cap on bytes the guest may hand the host in a single capability call.
    pub max_request_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            fuel: 1_000_000_000,
            wall_clock_ms: 120_000,
            memory_bytes: 64 * 1024 * 1024,
            table_elements: 10_000,
            max_steps: 8,
            max_request_bytes: 4 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    /// Call the Anthropic Messages API.
    Anthropic,
    /// Deterministic offline stand-in, for tests and demos without a key.
    Mock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl Effort {
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Thinking {
    Adaptive,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    pub provider: Provider,
    pub model: String,
    pub max_tokens: u32,
    pub effort: Effort,
    pub thinking: Thinking,
    /// Whether to ask for summarized reasoning. The default on Opus 5 is
    /// omitted, which streams empty thinking text.
    pub summarize_thinking: bool,
    pub base_url: String,
    pub api_version: String,
    /// Environment variable holding the API key. The key itself never appears
    /// in the policy file, and never crosses into the guest.
    pub api_key_env: String,
    pub request_timeout_ms: u64,
    pub max_retries: u32,
    /// Ask the API to re-run a policy-declined request on a recommended
    /// fallback model inside the same call. Without it, a refusal simply stops.
    pub server_side_fallback: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig {
            provider: Provider::Anthropic,
            model: "claude-opus-5".to_string(),
            max_tokens: 8192,
            effort: Effort::High,
            thinking: Thinking::Adaptive,
            summarize_thinking: false,
            base_url: "https://api.anthropic.com".to_string(),
            api_version: "2023-06-01".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            request_timeout_ms: 600_000,
            max_retries: 2,
            server_side_fallback: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolsConfig {
    /// Allowlist. A tool not named here does not exist as far as the agent knows.
    pub enabled: Vec<String>,
    #[serde(default)]
    pub http: HttpToolConfig,
    #[serde(default)]
    pub kv: KvToolConfig,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig {
            enabled: vec![
                "now".to_string(),
                "kv_get".to_string(),
                "kv_put".to_string(),
            ],
            http: HttpToolConfig::default(),
            kv: KvToolConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpToolConfig {
    /// Hosts the agent may fetch. Empty means the tool can reach nothing, even
    /// when it is listed in `enabled`.
    pub allowed_hosts: Vec<String>,
    pub max_response_bytes: usize,
    pub timeout_ms: u64,
}

impl Default for HttpToolConfig {
    fn default() -> Self {
        HttpToolConfig {
            allowed_hosts: Vec::new(),
            max_response_bytes: 64 * 1024,
            timeout_ms: 15_000,
        }
    }
}

impl HttpToolConfig {
    /// Exact host match, or a `*.example.com` suffix match. Never a substring
    /// match — `evil-example.com` must not pass a rule for `example.com`.
    pub fn allows(&self, host: &str) -> bool {
        let host = host.to_ascii_lowercase();
        self.allowed_hosts.iter().any(|rule| {
            let rule = rule.trim().to_ascii_lowercase();
            match rule.strip_prefix("*.") {
                Some(suffix) => host == suffix || host.ends_with(&format!(".{suffix}")),
                None => host == rule,
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvToolConfig {
    pub max_keys: usize,
    pub max_value_bytes: usize,
}

impl Default for KvToolConfig {
    fn default() -> Self {
        KvToolConfig {
            max_keys: 128,
            max_value_bytes: 16 * 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_rules_match_subdomains_but_not_lookalikes() {
        let cfg = HttpToolConfig {
            allowed_hosts: vec!["*.example.com".into(), "api.github.com".into()],
            ..Default::default()
        };
        assert!(cfg.allows("example.com"));
        assert!(cfg.allows("docs.example.com"));
        assert!(cfg.allows("API.GITHUB.COM"));
        // The lookalike is the whole point of not doing a substring match.
        assert!(!cfg.allows("evil-example.com"));
        assert!(!cfg.allows("example.com.evil.net"));
        assert!(!cfg.allows("github.com"));
    }

    #[test]
    fn an_empty_allowlist_reaches_nothing() {
        let cfg = HttpToolConfig::default();
        assert!(!cfg.allows("example.com"));
    }

    #[test]
    fn disabled_thinking_above_high_effort_is_rejected_at_load() {
        let mut policy = Policy::default();
        policy.llm.thinking = Thinking::Disabled;
        policy.llm.effort = Effort::Max;
        assert!(policy.validate().is_err());

        policy.llm.effort = Effort::High;
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn the_default_policy_is_valid() {
        Policy::default().validate().unwrap();
    }
}
