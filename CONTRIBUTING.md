# Contributing

This is a personal project, but if you're poking at it, here's how it's set up.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Rules I hold myself to

- Correctness before speed. Every kernel gets checked against a CPU reference
  before it gets optimised.
- No benchmark number goes in the README until it's actually been measured.
- Every commit builds and runs.