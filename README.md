# drift-linalg

**Drift-free linear algebra primitives for deterministic physics.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Built on [drift-kernel](https://github.com/aduboseh/drift-kernel)'s Neumaier-compensated summation.

## What This Solves

Standard floating-point accumulation drifts over time:
- Position integrators accumulate error
- Energy counters leak precision
- Long simulations diverge from expected values

`drift-linalg` provides spatial types that maintain O(ε) bounded error regardless of operation count.

## Usage

```rust
use drift_linalg::{Vec3, Vec3Accumulator};

let mut position = Vec3Accumulator::new();
let velocity = Vec3::new(1.0, 2.0, 3.0);

// Integrate over many frames without drift
for _ in 0..100_000 {
    position.add_scaled(velocity, 1.0 / 60.0);
}

let final_pos = position.resolve();
```

## Types

- `Vec3` — Standard 3D vector (f64 components)
- `Vec3Accumulator` — Drift-free 3D accumulator

## Features

- `serialization` — Enable serde support (optional)

## Serialization for Determinism

Use `to_le_bytes()` / `from_le_bytes()` for determinism verification:

```rust
let state = position.resolve();
let bytes = state.to_le_bytes();  // 24 bytes, platform-consistent
let hash = sha256(&bytes);
```

Do NOT use text formatting for hashing — floating-point text representation is not platform-consistent.

## License

Apache-2.0
