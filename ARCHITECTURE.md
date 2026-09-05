# verbum.cpp — Architecture

## What this is

An LLM inference engine written from scratch in C++ and CUDA. No PyTorch, no
llama.cpp, no third-party runtime doing the actual math -- it loads published
open weights (Qwen3-0.6B) and runs the entire forward pass, from tokenizer to
sampled output, in code written for this project.

Three layers sit on top of the core engine: INT8 quantization, a CUDA
backend, and an NPC memory demo built on Lattice, a vector database also
built from scratch. All three are optional, switched on independently, and
none of them change the core engine's behavior when switched off.

## Correctness strategy

Every piece of math in this engine was checked against an independent
reference before being trusted, in this order of increasing rigor:

1. **Hand-computed values** for the smallest possible cases (a 2x2 matmul,
   a single RMSNorm row) -- catches gross indexing errors immediately.
2. **A numpy reimplementation** of the same formulas, for every kernel
   individually (RMSNorm, RoPE, attention, SwiGLU). Catches C++-specific
   bugs -- off-by-ones, wrong loop bounds -- but not a shared
   misunderstanding of the algorithm, since the same mistake could exist in
   both implementations.
3. **Real HuggingFace `transformers` output**, compared directly. The whole
   forward pass, real weights, three prompts, logits compared position by
   position. This is the actual gate: argmax matched on all three, max
   logit difference ~3e-5 (floating-point noise from bf16 widening and
   accumulation order, not a bug).
4. **Structural equivalence tests** for anything with two code paths that
   should produce identical results -- the KV-cache vs. full-sequence
   attention (0.000000 diff, not "close"), and CPU vs. CUDA vs. quantized
   generation (identical generated text on the same prompt).

Two specific bugs this process caught before they shipped, worth naming
because they're the kind that don't crash:

- **RoPE has two valid rotation conventions** (the paper's adjacent-pair
  form, HuggingFace's split-half form). Both compile, both run, only one
  matches how the weights were trained. Testing only at position 0 can't
  catch this -- position 0 is the identity rotation under either
  convention. A real position must be tested.
- **Grouped-query attention's head-to-kv-head mapping** has a wrong version
  (`h % KVH`) that looks as plausible as the right one (`h / group`), and a
  test with only one KV head can't distinguish them, since every mapping
  collapses to the same thing when there's nothing to map between.

## Core engine (`core/`)

- **`tensor.{h,cpp}`** -- a plain, owning, row-major float32 tensor. No
  views, no broadcasting, deliberately simple so it's easy to be certain
  it's correct.
- **`safetensors.{h,cpp}`** -- mmaps the weight file directly, parses the
  JSON header, hands out pointers into the mapped region. bf16 and f16 are
  widened to f32 on read.
- **`tokenizer.{h,cpp}`** -- byte-level BPE, matching HuggingFace's
  tokenizer exactly (verified against 13 reference cases covering
  whitespace, digits, unicode, and special tokens).
- **`ops.{h,cpp}`** -- naive and tiled CPU matmul.
- **`nn.{h,cpp}`** -- RMSNorm, RoPE (split-half convention), embedding
  lookup.
- **`attention.{h,cpp}`** -- QK-Norm, grouped-query attention, causal
  masking. QK-Norm is Qwen3-specific -- RMSNorm applied per-head to Q and K
  before RoPE, not part of vanilla LLaMA-style attention.
- **`model.{h,cpp}`** -- ties the above into the full transformer stack:
  embedding -> N layers (pre-norm, attention, residual, pre-norm, SwiGLU
  FFN, residual) -> final norm -> lm_head. Exposes both a batched
  `forward()` and a cached, one-token-at-a-time `step()`.
- **`generate.{h,cpp}`** -- the KV-cache, and a sampler (greedy, top-k,
  top-p, temperature).

## Quantization (`core/src/quant.cpp`)

Per-row symmetric INT8 quantization of the attention and FFN projection
matrices. Per-row rather than per-tensor because output channels genuinely
differ in magnitude -- measured directly on synthetic weights with a
deliberate 50x spread: per-row gives 0.80% relative RMS error, a single
global scale gives 1.33%.

Real measurement on Qwen3-0.6B's actual weights: 1761.61 MB -> 441.78 MB on
the quantized matrices (3.99x), worst tensor at 1.20% error, all 196
quantized tensors under 1.3%.

**Known limitation, disclosed rather than hidden:** the embedding table and
`lm_head` stay f32 and dominate whole-model memory -- embed_tokens alone is
622 MB, larger than any other single tensor. The honest whole-model ratio
is 1.78x as currently coded (there's also an unrelated pre-existing bug
where `lm_head` is loaded as a full duplicate of `embed_tokens` even though
the model ties them -- fixed cost would bring the ratio to 2.24x). Verified:
identical generated output between f32 and int8 weights on the same prompt.
Quantization's speed effect varied more across machines than expected
(34-44% faster decode) rather than the near-flat result first measured --
plausible mechanism is that int8 moves 4x less data through memory even
though the arithmetic itself isn't vectorized, and decode is a
memory-bandwidth-bound workload. Treated as an honest observation with a
plausible cause, not a settled multiplier, since every measurement here is
a single run.

## CUDA backend (`kernels/cuda_ops.cu`, behind `VERBUM_ENABLE_CUDA`)

Off by default -- the default build has no CUDA dependency at all. When
enabled, weights are uploaded to device memory once at model load (not
per-token), and `step()` dispatches to a GPU decode path that mirrors the
CPU path's structure exactly.

Verified identical generated text between CPU and CUDA on a Tesla T4.
Same-session comparison: 28.01 tok/s CUDA vs. 0.73 tok/s CPU on that
Kaggle instance (38.4x) -- but Kaggle's shared CPU allocation has shown
2x+ swings session to session, so the more durable comparison is against
this project's actual development machine: 1.83 tok/s, giving CUDA a
14.7-38x range depending which CPU baseline it's measured against. The GPU
number itself is the reproducible one -- two separate sessions on
different days landed within 5% of each other (26.83 and 28.01 tok/s).

Sanity-checked against hardware: a T4's memory bandwidth (~320 GB/s) puts a
theoretical decode ceiling around 106 tok/s. ~26-28 tok/s sits at roughly
25% of that ceiling -- expected for a first integration with per-operation
kernel launches and no kernel fusion, real room for further optimization.

**Known limitation:** quantization and CUDA are not combined yet.

## Python layer (`bindings/`, `app/`)

`bindings/verbum_py.cpp` wraps the engine as a pybind11 module exposing
text in, text out. Both `generate()` and the `Engine` constructor release
the GIL during their call (`py::call_guard<py::gil_scoped_release>()`).

**A real bug caught during the demo build, worth stating plainly:**
`generate()` had the GIL release from the start, but the constructor
initially didn't. Since model construction does real, multi-second C++
work (mmap, per-layer materialization), *any* other Python thread --
including a UI event loop -- was completely frozen for the entire load,
regardless of whether that load happened on a background thread. Confirmed
directly: before the fix, a Pygame loading screen drew exactly 1 frame in
0.58 seconds; after adding the guard to the constructor too, the same load
drew 144 frames in 4.97 seconds, matching the intended 30fps cap almost
exactly. The lesson: releasing the GIL on the "obviously slow" call isn't
enough if another call that also does real native work is holding it.

`app/npc.py` is the NPC orchestration layer: persona strings, per-NPC
memory, and prompt assembly, deliberately kept in Python.

`app/lattice_index.py` adapts Lattice's real API to the small interface
the NPC memory layer expects -- `insert(id, vec)`, `query(vec, k)`.

**`embed_text()`** produces the vectors NPC memories are stored and
searched by: mean-pooled, L2-normalized final hidden states from the
engine itself. Verified to correctly rank a related pair of sentences
above an unrelated one, though the margin is thin (~0.05) -- a known
property of raw decoder hidden states, sufficient for a demo with a
handful of memories per NPC.

**Qwen3 is a hybrid reasoning model** and emits a `<think>...</think>`
block by default. The prompt template pre-fills an empty, already-closed
think block right after the assistant turn starts, suppressing reasoning
output so NPCs speak in character instead of narrating their own thought
process.

**A second real bug, caught by a crash rather than a code review:**
Lattice's vector storage genuinely persists across separate demo launches
(it's a real WAL-backed database), but `MemoryRecord`'s text side never
did -- it only ever lived in an in-memory Python list that started fresh,
re-numbering ids from zero, every time the process restarted. A fresh
session's locally-numbered ids could collide with old ids still sitting in
Lattice's on-disk index from a previous run, and a query could return an
id with no corresponding text anywhere in the current session --
`IndexError: list index out of range`. The fix: wipe `npc_data/` at every
launch, so both storage layers start honestly in sync. Cross-session NPC
memory was never actually a tested or intended feature of this demo --
memory persists correctly within one continuous run, which is what every
test in this project has actually verified.

## Demo shell (`app/demo.py`)

A Pygame scene: two NPCs (Meera, a blacksmith; Arjun, an innkeeper), click
to talk, a text input with clipboard paste (via `pyperclip`, not Pygame's
own `scrap` module, which is unreliable on macOS) and a blinking cursor, a
reply bubble, and a visible indicator of what the NPC recalled for that
turn.

Generation and model loading both run on background threads
(`app/async_npc.py` and an equivalent inline pattern for startup),
communicating back to the render loop through polling, so the window stays
responsive rather than being flagged "Not Responding" by the OS during
multi-second native calls.

**Personas explicitly instruct the model not to echo the player's own
sentence back to them, even partly.** Without this, a small model's
easiest path to a plausible-sounding reply was often to lightly reword the
player's own line -- which, combined with first-person phrasing like "my
brother," produced replies that sounded like the NPC was confused about
whose sibling had gone missing. The fix isn't a perspective correction so
much as removing the incentive to echo in the first place.

Only one generation runs at a time across the whole demo, deliberately --
not per NPC. Both NPCs share one loaded model instance; running two
generations concurrently would mean two threads mutating that model's
internal state at once, a real data race. The UI reflects this state
honestly ("X is still responding...") rather than hiding it.

## Build configuration

| Flag | Default | Effect |
|---|---|---|
| `VERBUM_ENABLE_CUDA` | OFF | Compiles `kernels/cuda_ops.cu`, defines `VERBUM_CUDA` |
| `VERBUM_BUILD_PYTHON` | OFF | Builds the pybind11 module |

`generate`'s CLI flags: `--quantize`, `--cuda` (mutually exclusive for
now), `--greedy`.

## Known limitations, stated plainly

- Prefill is unbatched -- one token at a time through the same path as
  decode, not the batched `forward()`.
- Quantization and CUDA don't compose.
- `lm_head` is loaded as a redundant full copy of `embed_tokens` despite
  the model tying them -- ~622 MB wasted, not yet fixed.
- NPC memory embeddings come from the engine's own hidden states, not a
  purpose-trained embedding model.
- NPC memory persists within one demo session, not across separate
  launches -- by design, given the mismatch described above.
- The demo shell is single-machine, local-only, and intentionally minimal.
