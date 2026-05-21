$ErrorActionPreference = "Stop"
cargo test --manifest-path rust/inditok-core/Cargo.toml
maturin develop
pytest

