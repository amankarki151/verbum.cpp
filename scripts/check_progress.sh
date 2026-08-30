#!/usr/bin/env bash
# Run from the project root: bash scripts/check_progress.sh
set -uo pipefail
cd "$(dirname "$0")/.."

pass=0
fail=0

ok()   { echo "  ok      $1"; }
miss() { echo "  MISSING $1"; fail=$((fail+1)); }
have() { [ -f "$1" ] && ok "$1" || miss "$1"; }

echo "=== Day 0/1/2 files ==="
have core/include/verbum/config.h
have core/include/verbum/safetensors.h
have core/include/verbum/tokenizer.h
have core/include/verbum/tensor.h
have core/include/verbum/ops.h
have core/src/config.cpp
have core/src/safetensors.cpp
have core/src/tokenizer.cpp
have core/src/tensor.cpp
have core/src/ops.cpp
have tests/test_safetensors.cpp
have tests/test_tokenizer.cpp
have tests/test_ops.cpp
have scripts/dump_reference_tokens.py
have tests/reference_tokens.json
have models/qwen3-0.6b/config.json
have models/qwen3-0.6b/tokenizer.json
have core/include/verbum/nn.h
have core/src/nn.cpp
have tests/test_nn.cpp
have scripts/dump_reference_nn.py
have tests/reference_nn.json
have core/include/verbum/attention.h
have core/src/attention.cpp
have tests/test_attention.cpp
have scripts/dump_reference_attn.py
have tests/reference_attn.json

echo ""
echo "=== build ==="
if cmake -B build -DCMAKE_BUILD_TYPE=Debug > /tmp/vb_configure.log 2>&1; then
  ok "cmake configure"
else
  miss "cmake configure (see /tmp/vb_configure.log)"
fi

if cmake --build build -j > /tmp/vb_build.log 2>&1; then
  ok "cmake build"
else
  miss "cmake build (see /tmp/vb_build.log)"
  tail -20 /tmp/vb_build.log
fi

echo ""
echo "=== tests ==="
run_test() {
  local name="$1"; shift
  local bin="build/$name"
  if [ ! -x "$bin" ]; then
    echo "  --      $name (not built yet)"
    return
  fi
  if "$bin" "$@" > "/tmp/vb_${name}.log" 2>&1; then
    ok "$name"
    pass=$((pass+1))
  else
    miss "$name (see /tmp/vb_${name}.log)"
    tail -15 "/tmp/vb_${name}.log"
  fi
}

run_test test_safetensors models/qwen3-0.6b
run_test test_tokenizer models/qwen3-0.6b tests/reference_tokens.json
run_test test_ops
run_test test_nn tests/reference_nn.json
run_test test_attention tests/reference_attn.json

echo ""
echo "=== git ==="
echo "commits:"
git log --oneline 2>/dev/null || echo "  (no commits yet)"
echo ""
echo "uncommitted changes:"
git status --short

echo ""
echo "=== summary ==="
echo "$pass test(s) passed, $fail check(s) failed"
[ "$fail" -eq 0 ] && echo "everything checked out clean" || echo "something above needs attention"