//! A containerized WebAssembly runtime that encapsulates an AI agent.
//!
//! The agent is compiled to a `wasm32-unknown-unknown` module with no WASI and
//! no syscalls. It reaches the outside world only through host functions this
//! crate provides, each checked against a [`policy::Policy`] the guest cannot
//! read or influence.
//!
//! ```no_run
//! use tds_host::{policy::Policy, runtime::Runtime};
//! use tds_abi::AgentInput;
//!
//! # fn main() -> anyhow::Result<()> {
//! let runtime = Runtime::from_files(Policy::default(), "agent.wasm".as_ref())?;
//! let report = runtime.run(AgentInput {
//!     task: "Summarize the release notes.".into(),
//!     context: None,
//!     max_steps: 4,
//! })?;
//! println!("{}", report.output.answer);
//! # Ok(())
//! # }
//! ```

pub mod llm;
pub mod policy;
pub mod runtime;
pub mod tools;
