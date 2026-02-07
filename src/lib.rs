//! # u-optim
//!
//! Mathematical primitives for the U-Engine ecosystem.
//!
//! This crate provides foundational mathematical, statistical, and probabilistic
//! building blocks that are domain-agnostic. It knows nothing about scheduling,
//! nesting, geometry, or any consumer domain.
//!
//! ## Modules
//!
//! - [`stats`] — Descriptive statistics with numerical stability guarantees
//!
//! ## Design Philosophy
//!
//! - **Numerical stability first**: Welford's algorithm for variance,
//!   Kahan summation for accumulation
//! - **No unnecessary dependencies**: Pure Rust for core math
//! - **Property-based testing**: Mathematical invariants verified via proptest

pub mod stats;
