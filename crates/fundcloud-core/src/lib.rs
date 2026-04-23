//! Fundcloud numerical core.
//!
//! Pure-Rust numerical kernels. No Python dependency lives here so the crate
//! can be benchmarked, fuzzed, and unit-tested in isolation. The Python
//! bindings live in the `fundcloud-py` crate.

#![deny(clippy::all)]
#![warn(missing_docs)]

use thiserror::Error;

pub mod drawdown;
pub mod moments;
pub mod returns;
pub mod rolling;
pub mod sim;
pub mod tail_risk;

/// Errors that numerical kernels may surface to the caller.
#[derive(Debug, Error)]
pub enum CoreError {
    /// Input array had a shape the kernel cannot accept.
    #[error("invalid shape: {0}")]
    Shape(String),
}

/// Build-time version string exposed for smoke testing.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Cheap no-op used by Python tests to verify the bindings reach this crate.
pub fn kernel_version() -> &'static str {
    CORE_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_not_empty() {
        assert!(!kernel_version().is_empty());
    }
}
