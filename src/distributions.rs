//! Probability distributions.
//!
//! Domain-agnostic probability distribution types with analytical
//! moments (mean, variance) and CDF/inverse-CDF evaluation.
//!
//! # Supported Distributions
//!
//! | Distribution | Parameters | Mean | Variance |
//! |---|---|---|---|
//! | [`Uniform`] | min, max | (a+b)/2 | (b−a)²/12 |
//! | [`Triangular`] | min, mode, max | (a+b+c)/3 | (a²+b²+c²−ab−ac−bc)/18 |
//! | [`Normal`] | μ, σ | μ | σ² |
//! | [`LogNormal`] | μ, σ | exp(μ+σ²/2) | (exp(σ²)−1)·exp(2μ+σ²) |
//! | [`Pert`] | min, mode, max | (a+4m+b)/6 | see docs |
//!
//! # Design Notes
//!
//! This module is **domain-agnostic**. There is no concept of "duration",
//! "scheduling", or any consumer domain. Parameters are plain `f64` values.

use crate::special;

/// Error type for invalid distribution parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionError {
    /// Parameters violate distribution constraints.
    InvalidParameters(String),
}

impl std::fmt::Display for DistributionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributionError::InvalidParameters(msg) => {
                write!(f, "invalid distribution parameters: {msg}")
            }
        }
    }
}

impl std::error::Error for DistributionError {}

// ============================================================================
// Uniform Distribution
// ============================================================================

/// Continuous uniform distribution on `[min, max]`.
///
/// # Mathematical Definition
/// - PDF: f(x) = 1/(max−min) for x ∈ [min, max]
/// - CDF: F(x) = (x−min)/(max−min)
/// - Mean: (min+max)/2
/// - Variance: (max−min)²/12
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    min: f64,
    max: f64,
}

impl Uniform {
    /// Creates a new uniform distribution on `[min, max]`.
    ///
    /// # Errors
    /// Returns `Err` if `min >= max` or either parameter is not finite.
    pub fn new(min: f64, max: f64) -> Result<Self, DistributionError> {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(DistributionError::InvalidParameters(format!(
                "Uniform requires min < max, got min={min}, max={max}"
            )));
        }
        Ok(Self { min, max })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn mean(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    pub fn variance(&self) -> f64 {
        let range = self.max - self.min;
        range * range / 12.0
    }

    /// CDF: F(x) = (x−min)/(max−min), clamped to [0, 1].
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            0.0
        } else if x >= self.max {
            1.0
        } else {
            (x - self.min) / (self.max - self.min)
        }
    }

    /// Inverse CDF (quantile function): x = min + p·(max−min).
    ///
    /// Returns `None` if `p` is outside `[0, 1]`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        Some(self.min + p * (self.max - self.min))
    }

    /// PDF: f(x) = 1/(max−min) for x ∈ [min, max], 0 otherwise.
    pub fn pdf(&self, x: f64) -> f64 {
        if x >= self.min && x <= self.max {
            1.0 / (self.max - self.min)
        } else {
            0.0
        }
    }
}

// ============================================================================
// Triangular Distribution
// ============================================================================

/// Triangular distribution with parameters `[min, mode, max]`.
///
/// # Mathematical Definition
/// - PDF: piecewise linear, peaking at mode
/// - CDF: piecewise quadratic
/// - Mean: (min + mode + max) / 3
/// - Variance: (a² + b² + c² − ab − ac − bc) / 18
///
/// Reference: Johnson, Kotz & Balakrishnan (1995), *Continuous Univariate
/// Distributions*, Vol. 2, Chapter 26.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangular {
    min: f64,
    mode: f64,
    max: f64,
}

impl Triangular {
    /// Creates a new triangular distribution.
    ///
    /// # Errors
    /// Returns `Err` if `min >= max` or `mode` is outside `[min, max]`.
    pub fn new(min: f64, mode: f64, max: f64) -> Result<Self, DistributionError> {
        if !min.is_finite() || !mode.is_finite() || !max.is_finite() {
            return Err(DistributionError::InvalidParameters(
                "Triangular parameters must be finite".into(),
            ));
        }
        if min > mode || mode > max || min >= max {
            return Err(DistributionError::InvalidParameters(format!(
                "Triangular requires min ≤ mode ≤ max and min < max, got {min}, {mode}, {max}"
            )));
        }
        Ok(Self { min, mode, max })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn mode(&self) -> f64 {
        self.mode
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    /// Mean = (min + mode + max) / 3.
    pub fn mean(&self) -> f64 {
        (self.min + self.mode + self.max) / 3.0
    }

    /// Variance = (a² + b² + c² − ab − ac − bc) / 18.
    pub fn variance(&self) -> f64 {
        let (a, b, c) = (self.min, self.mode, self.max);
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }

    /// PDF of the triangular distribution.
    ///
    /// ```text
    /// f(x) = 2(x−a) / ((c−a)(b−a))  for a ≤ x ≤ b
    ///      = 2(c−x) / ((c−a)(c−b))  for b < x ≤ c
    ///      = 0                       otherwise
    /// ```
    pub fn pdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.min, self.mode, self.max);
        if x < a || x > c {
            0.0
        } else if x <= b {
            2.0 * (x - a) / ((c - a) * (b - a).max(f64::MIN_POSITIVE))
        } else {
            2.0 * (c - x) / ((c - a) * (c - b).max(f64::MIN_POSITIVE))
        }
    }

    /// CDF of the triangular distribution.
    ///
    /// ```text
    /// F(x) = (x−a)² / ((c−a)(b−a))       for a ≤ x ≤ b
    ///      = 1 − (c−x)² / ((c−a)(c−b))   for b < x ≤ c
    /// ```
    pub fn cdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.min, self.mode, self.max);
        if x <= a {
            0.0
        } else if x <= b {
            (x - a) * (x - a) / ((c - a) * (b - a).max(f64::MIN_POSITIVE))
        } else if x < c {
            1.0 - (c - x) * (c - x) / ((c - a) * (c - b).max(f64::MIN_POSITIVE))
        } else {
            1.0
        }
    }

    /// Inverse CDF (quantile function) of the triangular distribution.
    ///
    /// ```text
    /// F⁻¹(p) = a + √(p·(c−a)·(b−a))                 if p < F(b)
    ///        = c − √((1−p)·(c−a)·(c−b))              if p ≥ F(b)
    /// ```
    ///
    /// Returns `None` if `p` is outside `[0, 1]`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&p) {
            return None;
        }
        let (a, b, c) = (self.min, self.mode, self.max);
        let fc = (b - a) / (c - a); // CDF at the mode
        if p < fc {
            Some(a + ((c - a) * (b - a) * p).sqrt())
        } else {
            Some(c - ((c - a) * (c - b) * (1.0 - p)).sqrt())
        }
    }
}

// ============================================================================
// Normal Distribution
// ============================================================================

/// Normal (Gaussian) distribution N(μ, σ²).
///
/// # Mathematical Definition
/// - PDF: φ(x) = (1/(σ√(2π))) exp(−(x−μ)²/(2σ²))
/// - CDF: Φ((x−μ)/σ) (via standard normal CDF)
/// - Mean: μ
/// - Variance: σ²
#[derive(Debug, Clone, PartialEq)]
pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    /// Creates a new normal distribution N(μ, σ).
    ///
    /// # Errors
    /// Returns `Err` if `sigma ≤ 0` or parameters are not finite.
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "Normal requires finite μ and σ > 0, got μ={mu}, σ={sigma}"
            )));
        }
        Ok(Self { mu, sigma })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn mean(&self) -> f64 {
        self.mu
    }

    pub fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    pub fn std_dev(&self) -> f64 {
        self.sigma
    }

    /// PDF: (1/(σ√(2π))) exp(−(x−μ)²/(2σ²)).
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        special::standard_normal_pdf(z) / self.sigma
    }

    /// CDF: Φ((x−μ)/σ).
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        special::standard_normal_cdf(z)
    }

    /// Inverse CDF (quantile): μ + σ·Φ⁻¹(p).
    ///
    /// Returns `None` if `p` is outside `(0, 1)`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        Some(self.mu + self.sigma * special::inverse_normal_cdf(p))
    }
}

// ============================================================================
// LogNormal Distribution
// ============================================================================

/// Log-normal distribution: if X ~ LogNormal(μ, σ), then ln(X) ~ N(μ, σ²).
///
/// # Mathematical Definition
/// - PDF: (1/(xσ√(2π))) exp(−(ln(x)−μ)²/(2σ²)) for x > 0
/// - CDF: Φ((ln(x)−μ)/σ)
/// - Mean: exp(μ + σ²/2)
/// - Variance: (exp(σ²) − 1) · exp(2μ + σ²)
///
/// Reference: Johnson, Kotz & Balakrishnan (1994), *Continuous Univariate
/// Distributions*, Vol. 1, Chapter 14.
#[derive(Debug, Clone, PartialEq)]
pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    /// Creates a new log-normal distribution.
    ///
    /// Parameters `mu` and `sigma` are the mean and std dev of ln(X).
    ///
    /// # Errors
    /// Returns `Err` if `sigma ≤ 0` or parameters are not finite.
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "LogNormal requires finite μ and σ > 0, got μ={mu}, σ={sigma}"
            )));
        }
        Ok(Self { mu, sigma })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Mean = exp(μ + σ²/2).
    pub fn mean(&self) -> f64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp()
    }

    /// Variance = (exp(σ²) − 1) · exp(2μ + σ²).
    pub fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (s2.exp() - 1.0) * (2.0 * self.mu + s2).exp()
    }

    /// PDF for x > 0.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let ln_x = x.ln();
        let z = (ln_x - self.mu) / self.sigma;
        special::standard_normal_pdf(z) / (x * self.sigma)
    }

    /// CDF: Φ((ln(x)−μ)/σ) for x > 0.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        special::standard_normal_cdf(z)
    }

    /// Inverse CDF: exp(μ + σ·Φ⁻¹(p)).
    ///
    /// Returns `None` if `p` is outside `(0, 1)`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        Some((self.mu + self.sigma * special::inverse_normal_cdf(p)).exp())
    }
}

// ============================================================================
// PERT Distribution (Modified Beta)
// ============================================================================

/// PERT distribution (Program Evaluation and Review Technique).
///
/// A modified Beta distribution defined by three points: optimistic (min),
/// most likely (mode), and pessimistic (max).
///
/// # Mathematical Definition
///
/// Shape parameters (with λ = 4):
/// ```text
/// α = 1 + λ · (mode − min) / (max − min)
/// β = 1 + λ · (max − mode) / (max − min)
/// ```
///
/// The underlying variable Y = (X − min)/(max − min) follows Beta(α, β).
///
/// - Mean: (min + λ·mode + max) / (λ + 2) = (min + 4·mode + max) / 6
/// - Std Dev (simplified): (max − min) / (λ + 2) = (max − min) / 6
///
/// # Exact vs Simplified Variance
///
/// The simplified variance `((max−min)/6)²` is an approximation. The exact
/// variance uses the Beta distribution formula:
/// ```text
/// Var = α·β / ((α+β)²·(α+β+1)) × (max−min)²
/// ```
///
/// The simplified formula is exact when the distribution is symmetric
/// (mode = midpoint) and becomes less accurate as skewness increases.
///
/// Reference: Malcolm et al. (1959), "Application of a Technique for
/// Research and Development Program Evaluation", *Operations Research* 7(5).
#[derive(Debug, Clone, PartialEq)]
pub struct Pert {
    min: f64,
    mode: f64,
    max: f64,
    alpha: f64,
    beta: f64,
}

impl Pert {
    /// Creates a standard PERT distribution (λ = 4).
    ///
    /// # Errors
    /// Returns `Err` if `min >= max` or `mode` is outside `[min, max]`.
    pub fn new(min: f64, mode: f64, max: f64) -> Result<Self, DistributionError> {
        Self::with_shape(min, mode, max, 4.0)
    }

    /// Creates a modified PERT distribution with custom shape parameter λ.
    ///
    /// λ controls the weight of the mode:
    /// - λ = 4: standard PERT
    /// - λ > 4: tighter distribution (more peaked)
    /// - λ < 4: flatter distribution (less peaked)
    ///
    /// # Errors
    /// Returns `Err` if parameters are invalid.
    pub fn with_shape(
        min: f64,
        mode: f64,
        max: f64,
        lambda: f64,
    ) -> Result<Self, DistributionError> {
        if !min.is_finite() || !mode.is_finite() || !max.is_finite() || !lambda.is_finite() {
            return Err(DistributionError::InvalidParameters(
                "PERT parameters must be finite".into(),
            ));
        }
        if min > mode || mode > max || min >= max {
            return Err(DistributionError::InvalidParameters(format!(
                "PERT requires min ≤ mode ≤ max and min < max, got {min}, {mode}, {max}"
            )));
        }
        if lambda <= 0.0 {
            return Err(DistributionError::InvalidParameters(format!(
                "PERT λ must be > 0, got {lambda}"
            )));
        }

        let range = max - min;
        let alpha = 1.0 + lambda * (mode - min) / range;
        let beta = 1.0 + lambda * (max - mode) / range;

        Ok(Self {
            min,
            mode,
            max,
            alpha,
            beta,
        })
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn mode(&self) -> f64 {
        self.mode
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn beta_param(&self) -> f64 {
        self.beta
    }

    /// Mean = (min + 4·mode + max) / 6 for standard PERT (λ=4).
    ///
    /// General: (min + λ·mode + max) / (λ + 2).
    pub fn mean(&self) -> f64 {
        let ab = self.alpha + self.beta;
        // Mean of Beta(α,β) on [min,max] = min + (α/(α+β))·(max-min)
        self.min + (self.alpha / ab) * (self.max - self.min)
    }

    /// Exact variance using Beta distribution formula.
    ///
    /// ```text
    /// Var = α·β / ((α+β)²·(α+β+1)) × (max−min)²
    /// ```
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        let range = self.max - self.min;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0)) * range * range
    }

    /// Standard deviation = √(variance).
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// CDF via regularized incomplete beta function approximation.
    ///
    /// Uses a numerical approximation of the regularized incomplete beta
    /// function I_x(α, β).
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }
        let t = (x - self.min) / (self.max - self.min);
        regularized_incomplete_beta(t, self.alpha, self.beta)
    }

    /// Approximate quantile using normal approximation.
    ///
    /// Uses `μ + σ·Φ⁻¹(p)` with the exact PERT mean and std dev.
    /// This is an approximation; accuracy decreases for highly skewed PERTs.
    ///
    /// Returns `None` if `p` is outside `(0, 1)`.
    pub fn quantile_approx(&self, p: f64) -> Option<f64> {
        if p <= 0.0 || p >= 1.0 {
            return None;
        }
        let z = special::inverse_normal_cdf(p);
        let result = self.mean() + z * self.std_dev();
        // Clamp to [min, max]
        Some(result.clamp(self.min, self.max))
    }
}

// ============================================================================
// Regularized Incomplete Beta Function
// ============================================================================

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the continued fraction representation (Lentz's method) for
/// numerical evaluation.
///
/// Reference: Press et al. (2007), *Numerical Recipes*, 3rd ed., §6.4.
///
/// # Accuracy
/// Relative error < 1e-10 for typical parameter ranges.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
    // Choose the form with better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b);

    // Continued fraction (Lentz's algorithm)
    let cf = beta_cf(x, a, b);

    (ln_prefix.exp() / a) * cf
}

/// Log of the Beta function: ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a+b).
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation of ln Γ(x).
///
/// Reference: Lanczos (1964), "A Precision Approximation of the Gamma
/// Function", *SIAM Journal on Numerical Analysis* 1(1).
///
/// # Accuracy
/// Relative error < 2 × 10⁻¹⁰ for x > 0.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients (g = 7)
    #[allow(clippy::excessive_precision)]
    const COEFFICIENTS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula: Γ(x)·Γ(1−x) = π/sin(πx)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (i, &c) in COEFFICIENTS[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Continued fraction for the incomplete beta function (Lentz's algorithm).
///
/// Reference: Press et al. (2007), *Numerical Recipes*, §6.4.
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-14;
    const TINY: f64 = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(TINY);
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;

        // Even step: d_{2m}
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 / (1.0 + num_even * d).max(TINY);
        c = (1.0 + num_even / c).max(TINY);
        h *= d * c;

        // Odd step: d_{2m+1}
        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 / (1.0 + num_odd * d).max(TINY);
        c = (1.0 + num_odd / c).max(TINY);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    h
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Uniform ---

    #[test]
    fn test_uniform_basic() {
        let u = Uniform::new(0.0, 10.0).unwrap();
        assert!((u.mean() - 5.0).abs() < 1e-15);
        assert!((u.variance() - 100.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_cdf() {
        let u = Uniform::new(0.0, 10.0).unwrap();
        assert_eq!(u.cdf(-1.0), 0.0);
        assert!((u.cdf(5.0) - 0.5).abs() < 1e-15);
        assert_eq!(u.cdf(11.0), 1.0);
    }

    #[test]
    fn test_uniform_quantile() {
        let u = Uniform::new(2.0, 8.0).unwrap();
        assert_eq!(u.quantile(0.0), Some(2.0));
        assert_eq!(u.quantile(1.0), Some(8.0));
        assert!((u.quantile(0.5).unwrap() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_uniform_pdf() {
        let u = Uniform::new(0.0, 5.0).unwrap();
        assert!((u.pdf(2.5) - 0.2).abs() < 1e-15);
        assert_eq!(u.pdf(-1.0), 0.0);
    }

    #[test]
    fn test_uniform_invalid() {
        assert!(Uniform::new(5.0, 5.0).is_err());
        assert!(Uniform::new(5.0, 3.0).is_err());
        assert!(Uniform::new(f64::NAN, 5.0).is_err());
    }

    // --- Triangular ---

    #[test]
    fn test_triangular_mean() {
        let t = Triangular::new(0.0, 3.0, 6.0).unwrap();
        assert!((t.mean() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_triangular_symmetric_variance() {
        let t = Triangular::new(0.0, 5.0, 10.0).unwrap();
        // Var = (0+100+25-0-0-50)/18 = 75/18 ≈ 4.1667
        let expected = (0.0 + 25.0 + 100.0 - 0.0 - 0.0 - 50.0) / 18.0;
        assert!((t.variance() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_cdf() {
        let t = Triangular::new(0.0, 5.0, 10.0).unwrap();
        assert!((t.cdf(0.0)).abs() < 1e-15);
        assert!((t.cdf(10.0) - 1.0).abs() < 1e-15);
        // At mode: F(5) = (5-0)²/((10-0)*(5-0)) = 25/50 = 0.5
        assert!((t.cdf(5.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_triangular_quantile() {
        let t = Triangular::new(0.0, 5.0, 10.0).unwrap();
        assert!((t.quantile(0.0).unwrap() - 0.0).abs() < 1e-15);
        assert!((t.quantile(1.0).unwrap() - 10.0).abs() < 1e-15);
        assert!((t.quantile(0.5).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangular_invalid() {
        assert!(Triangular::new(5.0, 3.0, 10.0).is_err()); // mode < min
        assert!(Triangular::new(0.0, 11.0, 10.0).is_err()); // mode > max
        assert!(Triangular::new(5.0, 5.0, 5.0).is_err()); // min == max
    }

    // --- Normal ---

    #[test]
    fn test_normal_standard() {
        let n = Normal::new(0.0, 1.0).unwrap();
        assert!((n.mean()).abs() < 1e-15);
        assert!((n.variance() - 1.0).abs() < 1e-15);
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_normal_shifted() {
        let n = Normal::new(10.0, 2.0).unwrap();
        assert!((n.mean() - 10.0).abs() < 1e-15);
        assert!((n.variance() - 4.0).abs() < 1e-15);
        assert!((n.cdf(10.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_normal_quantile() {
        let n = Normal::new(0.0, 1.0).unwrap();
        assert!((n.quantile(0.5).unwrap()).abs() < 0.01);
        assert!((n.quantile(0.975).unwrap() - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_normal_invalid() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
    }

    // --- LogNormal ---

    #[test]
    fn test_lognormal_mean() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        let expected = (0.5_f64).exp(); // exp(0 + 1/2)
        assert!((ln.mean() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_cdf() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        assert_eq!(ln.cdf(0.0), 0.0);
        // Median of LogNormal(0,1) = exp(0) = 1.0
        assert!((ln.cdf(1.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_lognormal_quantile() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        // Median = exp(μ) = 1.0
        let q50 = ln.quantile(0.5).unwrap();
        assert!((q50 - 1.0).abs() < 0.01);
    }

    // --- PERT ---

    #[test]
    fn test_pert_mean() {
        let p = Pert::new(1.0, 4.0, 7.0).unwrap();
        // Mean = (1 + 4*4 + 7) / 6 = 24/6 = 4.0
        assert!((p.mean() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pert_symmetric_variance() {
        let p = Pert::new(0.0, 5.0, 10.0).unwrap();
        // For symmetric PERT: α = β = 3, range = 10
        // Var = 3*3/(6*6*7) * 100 = 900/2520 * 100 ≈ 3.571
        let expected = 9.0 / (36.0 * 7.0) * 100.0;
        assert!(
            (p.variance() - expected).abs() < 1e-10,
            "PERT variance: {} vs expected: {}",
            p.variance(),
            expected
        );
    }

    #[test]
    fn test_pert_shape_params() {
        let p = Pert::new(0.0, 5.0, 10.0).unwrap();
        // α = 1 + 4*(5-0)/(10-0) = 1 + 2 = 3
        // β = 1 + 4*(10-5)/(10-0) = 1 + 2 = 3
        assert!((p.alpha() - 3.0).abs() < 1e-15);
        assert!((p.beta_param() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_pert_cdf_bounds() {
        let p = Pert::new(1.0, 4.0, 7.0).unwrap();
        assert_eq!(p.cdf(0.0), 0.0);
        assert_eq!(p.cdf(8.0), 1.0);
        // CDF at mean should be close to 0.5 for symmetric PERT
        let p_sym = Pert::new(0.0, 5.0, 10.0).unwrap();
        assert!((p_sym.cdf(5.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_pert_cdf_monotonic() {
        let p = Pert::new(0.0, 3.0, 10.0).unwrap();
        let mut prev = 0.0;
        for i in 0..=100 {
            let x = i as f64 * 0.1;
            let c = p.cdf(x);
            assert!(c >= prev - 1e-15, "CDF not monotonic at x={x}");
            prev = c;
        }
    }

    #[test]
    fn test_pert_quantile_approx() {
        let p = Pert::new(0.0, 5.0, 10.0).unwrap();
        let q50 = p.quantile_approx(0.5).unwrap();
        assert!((q50 - 5.0).abs() < 0.5, "median approx: {q50}");
    }

    #[test]
    fn test_pert_invalid() {
        assert!(Pert::new(5.0, 3.0, 10.0).is_err());
        assert!(Pert::new(5.0, 5.0, 5.0).is_err());
    }

    #[test]
    fn test_pert_with_lambda() {
        // Higher lambda = more peaked
        let p4 = Pert::with_shape(0.0, 5.0, 10.0, 4.0).unwrap();
        let p8 = Pert::with_shape(0.0, 5.0, 10.0, 8.0).unwrap();
        assert!(
            p8.variance() < p4.variance(),
            "higher λ should give lower variance"
        );
    }

    // --- Regularized Incomplete Beta Function ---

    #[test]
    fn test_regularized_beta_bounds() {
        assert_eq!(regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0);
        assert_eq!(regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0);
    }

    #[test]
    fn test_regularized_beta_symmetric() {
        // For Beta(a,a), I_{0.5}(a,a) = 0.5 by symmetry
        let result = regularized_incomplete_beta(0.5, 3.0, 3.0);
        assert!(
            (result - 0.5).abs() < 1e-8,
            "I_0.5(3,3) = {result}, expected 0.5"
        );
    }

    #[test]
    fn test_regularized_beta_known_values() {
        // I_x(1,1) = x (Uniform)
        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let result = regularized_incomplete_beta(x, 1.0, 1.0);
            assert!(
                (result - x).abs() < 1e-10,
                "I_{x}(1,1) = {result}, expected {x}"
            );
        }

        // I_x(1,b) = 1 - (1-x)^b
        for &x in &[0.1, 0.5, 0.9] {
            let result = regularized_incomplete_beta(x, 1.0, 3.0);
            let expected = 1.0 - (1.0 - x).powi(3);
            assert!(
                (result - expected).abs() < 1e-10,
                "I_{x}(1,3) = {result}, expected {expected}"
            );
        }
    }

    // --- ln_gamma ---

    #[test]
    fn test_ln_gamma_known() {
        // Γ(1) = 1, ln(1) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // Γ(2) = 1, ln(1) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // Γ(3) = 2, ln(2) ≈ 0.6931
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
        // Γ(5) = 24, ln(24) ≈ 3.1781
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
        // Γ(0.5) = √π
        assert!(
            (ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10,
            "ln Γ(0.5) = {}, expected {}",
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln()
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        // --- Uniform ---

        #[test]
        fn uniform_cdf_in_01(
            min in -100.0_f64..0.0,
            max in 1.0_f64..100.0,
            x in -200.0_f64..200.0,
        ) {
            let u = Uniform::new(min, max).unwrap();
            let c = u.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c));
        }

        #[test]
        fn uniform_quantile_roundtrip(
            min in -100.0_f64..0.0,
            max in 1.0_f64..100.0,
            p in 0.0_f64..=1.0,
        ) {
            let u = Uniform::new(min, max).unwrap();
            let x = u.quantile(p).unwrap();
            let p_back = u.cdf(x);
            prop_assert!((p_back - p).abs() < 1e-12, "roundtrip: p={p} -> x={x} -> p_back={p_back}");
        }

        // --- Triangular ---

        #[test]
        fn triangular_cdf_in_01(
            min in -100.0_f64..-1.0,
            mode_frac in 0.0_f64..=1.0,
            range in 1.0_f64..100.0,
            x in -200.0_f64..200.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let t = Triangular::new(min, mode, max).unwrap();
            let c = t.cdf(x);
            prop_assert!((0.0..=1.0).contains(&c));
        }

        #[test]
        fn triangular_quantile_roundtrip(
            min in -50.0_f64..0.0,
            mode_frac in 0.01_f64..0.99,
            range in 1.0_f64..50.0,
            p in 0.001_f64..0.999,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let t = Triangular::new(min, mode, max).unwrap();
            let x = t.quantile(p).unwrap();
            let p_back = t.cdf(x);
            prop_assert!(
                (p_back - p).abs() < 1e-8,
                "roundtrip: p={p} -> x={x} -> p_back={p_back}"
            );
        }

        // --- PERT ---

        #[test]
        fn pert_mean_formula(
            min in -50.0_f64..0.0,
            mode_frac in 0.01_f64..0.99,
            range in 1.0_f64..50.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let p = Pert::new(min, mode, max).unwrap();
            let expected = (min + 4.0 * mode + max) / 6.0;
            prop_assert!(
                (p.mean() - expected).abs() < 1e-10,
                "PERT mean: {} vs expected: {}",
                p.mean(),
                expected
            );
        }

        #[test]
        fn pert_variance_positive(
            min in -50.0_f64..0.0,
            mode_frac in 0.01_f64..0.99,
            range in 1.0_f64..50.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let p = Pert::new(min, mode, max).unwrap();
            prop_assert!(p.variance() > 0.0);
        }

        #[test]
        fn pert_cdf_monotonic(
            min in -50.0_f64..0.0,
            mode_frac in 0.05_f64..0.95,
            range in 2.0_f64..50.0,
        ) {
            let max = min + range;
            let mode = min + mode_frac * range;
            let p = Pert::new(min, mode, max).unwrap();
            let mut prev = 0.0;
            for i in 0..=20 {
                let x = min + (i as f64 / 20.0) * range;
                let c = p.cdf(x);
                prop_assert!(c >= prev - 1e-10, "CDF not monotonic at x={x}");
                prev = c;
            }
        }
    }
}
