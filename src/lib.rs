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
//! - [`distributions`] — Probability distributions (Uniform, Triangular, PERT, Normal, LogNormal)
//! - [`special`] — Special mathematical functions (normal CDF, inverse normal CDF)
//!
//! ## Design Philosophy
//!
//! - **Numerical stability first**: Welford's algorithm for variance,
//!   Neumaier summation for accumulation
//! - **No unnecessary dependencies**: Pure Rust for core math
//! - **Property-based testing**: Mathematical invariants verified via proptest

pub mod distributions;
pub mod special;
pub mod stats;
