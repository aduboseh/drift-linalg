# drift-linalg

**Compensated vector accumulation primitives for numerical simulations.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Built on [drift-kernel](https://github.com/aduboseh/drift-kernel)'s Neumaier-compensated summation.

## What This Solves

Standard floating-point accumulation drifts over time:
- Position integrators accumulate error
- Energy counters leak precision
- Long simulations diverge from expected values

`drift-linalg` provides `Vec3Accumulator`, which applies Neumaier compensated summation independently to each coordinate. This often reduces accumulated rounding error, but does not eliminate error or provide an operation-count-independent absolute bound. Ordinary `Vec3` arithmetic and `dot` are not compensated; `add_scaled` multiplies in standard `f64` before compensated accumulation.

## Usage

```rust
use drift_linalg::{Vec3, Vec3Accumulator};

let mut position = Vec3Accumulator::new();
let velocity = Vec3::new(1.0, 2.0, 3.0);

// Integrate over many frames with compensated accumulation
for _ in 0..100_000 {
    position.add_scaled(velocity, 1.0 / 60.0);
}

let final_pos = position.resolve();
```

## Types

- `Vec3` — Standard 3D vector (f64 components)
- `Vec3Accumulator` — Component-wise compensated 3D accumulator

## Features

- `serialization` — Enable serde support (optional)

## Serialization for Determinism

Use `to_le_bytes()` / `from_le_bytes()` for determinism verification:

```rust
let state = position.resolve();
let bytes = state.to_le_bytes();  // 24 bytes, platform-consistent
let hash = sha256(&bytes); // Also encode all other future-affecting state.
```

Raw little-endian bytes preserve each floating-point bit pattern, including signed zero and NaN payloads. This alone does not provide cross-architecture replay: execution ordering, input/state completeness, rounding behavior, and compiler configuration must be governed. Avoid human-readable float formatting as a canonical hash representation.

## Limitations

- `Vec3::dot`, `Add`, `Sub`, `scale`, and `magnitude` use ordinary floating-point operations.
- `Vec3Accumulator::add_scaled` only compensates the additions, not multiplication.
- Nonfinite values, overflow and ill-conditioned inputs have no finite-error guarantee.
- Determinism requires a separately governed runtime and a complete state snapshot.

## License

Apache-2.0
