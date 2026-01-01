//! Drift-Linalg — Drift-Free Linear Algebra Primitives
//!
//! Provides spatial accumulator types built on top of `drift-kernel`'s
//! Neumaier-compensated summation. These types allow physics simulations
//! to maintain bounded numerical error across millions of operations.
//!
//! # Usage
//!
//! ```rust
//! use drift_linalg::{Vec3, Vec3Accumulator};
//!
//! let mut position = Vec3Accumulator::new();
//! let velocity = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
//!
//! // Integrate over many frames without drift
//! for _ in 0..100_000 {
//!     position.add_scaled(velocity, 1.0 / 60.0);
//! }
//!
//! let final_pos = position.resolve();
//! ```
//!
//! # Serialization
//!
//! The `serialization` feature enables serde support. Without it, the crate
//! has zero dependencies beyond `drift-kernel`.

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use drift_kernel::Neumaier;

/// A standard 3D vector
///
/// This type is used for inputs and outputs. For accumulation across
/// many operations, use [`Vec3Accumulator`] instead.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    /// Create a new Vec3.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns the raw IEEE-754 little-endian bytes.
    ///
    /// This is the **only valid way** to hash state for determinism verification.
    /// Do NOT use text formatting (Debug, Display) for hashing—floating-point
    /// text representation is not guaranteed to be platform-consistent.
    #[inline]
    pub fn to_le_bytes(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..8].copy_from_slice(&self.x.to_le_bytes());
        buf[8..16].copy_from_slice(&self.y.to_le_bytes());
        buf[16..24].copy_from_slice(&self.z.to_le_bytes());
        buf
    }

    /// Reconstruct a Vec3 from little-endian bytes.
    ///
    /// This is the inverse of [`to_le_bytes`](Self::to_le_bytes) and is required
    /// for checkpoint restore and replay branching.
    #[inline]
    pub fn from_le_bytes(bytes: [u8; 24]) -> Self {
        Self {
            x: f64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            y: f64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            z: f64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        }
    }

    /// Compute the dot product with another vector.
    #[inline]
    pub fn dot(&self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Compute the squared magnitude (avoids sqrt).
    #[inline]
    pub fn magnitude_squared(&self) -> f64 {
        self.dot(*self)
    }

    /// Compute the magnitude.
    #[inline]
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Scale by a scalar.
    #[inline]
    pub fn scale(&self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// A 3D spatial accumulator
///
/// Uses Neumaier-compensated summation on each component to maintain
/// O(ε) bounded error regardless of operation count.
///
/// # Example
///
/// ```rust
/// use drift_linalg::{Vec3, Vec3Accumulator};
///
/// let mut acc = Vec3Accumulator::new();
///
/// // These would drift in standard floats
/// for _ in 0..100_000 {
///     acc.add(Vec3 { x: 1e15, y: 1e-15, z: 1.0 });
///     acc.add(Vec3 { x: -1e15, y: -1e-15, z: -1.0 });
/// }
///
/// let result = acc.resolve();
/// assert!(result.x.abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct Vec3Accumulator {
    x: Neumaier,
    y: Neumaier,
    z: Neumaier,
}

impl Vec3Accumulator {
    /// Create a new zero-initialized accumulator.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an accumulator with an initial value.
    #[inline]
    pub fn with_initial(initial: Vec3) -> Self {
        Self {
            x: Neumaier::new(initial.x),
            y: Neumaier::new(initial.y),
            z: Neumaier::new(initial.z),
        }
    }

    /// Add a vector to the accumulator.
    #[inline]
    pub fn add(&mut self, vec: Vec3) {
        self.x.add(vec.x);
        self.y.add(vec.y);
        self.z.add(vec.z);
    }

    /// Add a scaled vector to the accumulator.
    ///
    /// # Note on Compensation
    ///
    /// **The scalar multiplication is NOT compensated.** Only the accumulation
    /// into the internal state uses Neumaier summation. The multiplication
    /// `vec.x * scalar` happens in standard f64 arithmetic.
    ///
    /// This is standard practice in numerical integration and is acceptable
    /// for most physics simulations. If you require compensated multiplication,
    /// you must implement it externally.
    #[inline]
    pub fn add_scaled(&mut self, vec: Vec3, scalar: f64) {
        self.x.add(vec.x * scalar);
        self.y.add(vec.y * scalar);
        self.z.add(vec.z * scalar);
    }

    /// Resolve the accumulator to a standard Vec3.
    ///
    /// This extracts the compensated total from each component.
    #[inline]
    pub fn resolve(&self) -> Vec3 {
        Vec3 {
            x: self.x.total(),
            y: self.y.total(),
            z: self.z.total(),
        }
    }

    /// Reset the accumulator to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
        self.z.reset();
    }
}

impl Default for Vec3Accumulator {
    fn default() -> Self {
        Self {
            x: Neumaier::new(0.0),
            y: Neumaier::new(0.0),
            z: Neumaier::new(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_to_from_le_bytes_roundtrip() {
        let original = Vec3::new(1.5, -2.25, 3.125);
        let bytes = original.to_le_bytes();
        let restored = Vec3::from_le_bytes(bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn vec3_accumulator_basic() {
        let mut acc = Vec3Accumulator::new();
        acc.add(Vec3::new(1.0, 2.0, 3.0));
        acc.add(Vec3::new(4.0, 5.0, 6.0));
        let result = acc.resolve();
        assert!((result.x - 5.0).abs() < 1e-15);
        assert!((result.y - 7.0).abs() < 1e-15);
        assert!((result.z - 9.0).abs() < 1e-15);
    }

    #[test]
    fn vec3_accumulator_catastrophic_cancellation() {
        let mut acc = Vec3Accumulator::new();

        // This would fail with naive summation
        acc.add(Vec3::new(1e16, 1e16, 1e16));
        acc.add(Vec3::new(1.0, 1.0, 1.0));
        acc.add(Vec3::new(-1e16, -1e16, -1e16));

        let result = acc.resolve();
        assert!((result.x - 1.0).abs() < 1e-10, "x: expected 1.0, got {}", result.x);
        assert!((result.y - 1.0).abs() < 1e-10, "y: expected 1.0, got {}", result.y);
        assert!((result.z - 1.0).abs() < 1e-10, "z: expected 1.0, got {}", result.z);
    }

    #[test]
    fn vec3_accumulator_long_horizon() {
        let mut acc = Vec3Accumulator::new();

        // 100k balanced operations
        for i in 0..100_000 {
            let large = 1e15 + (i as f64) * 1e-5;
            acc.add(Vec3::new(large, large, large));
            acc.add(Vec3::new(-large, -large, -large));
        }

        let result = acc.resolve();
        assert!(result.x.abs() < 1e-10, "x drift: {}", result.x);
        assert!(result.y.abs() < 1e-10, "y drift: {}", result.y);
        assert!(result.z.abs() < 1e-10, "z drift: {}", result.z);
    }

    #[test]
    fn vec3_accumulator_add_scaled() {
        let mut acc = Vec3Accumulator::new();
        let vec = Vec3::new(10.0, 20.0, 30.0);
        acc.add_scaled(vec, 0.5);
        let result = acc.resolve();
        assert!((result.x - 5.0).abs() < 1e-15);
        assert!((result.y - 10.0).abs() < 1e-15);
        assert!((result.z - 15.0).abs() < 1e-15);
    }
}
