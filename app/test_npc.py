"""Tests the NPC logic against a fake engine and a brute-force index, so a
failure here is a logic bug, not a model or index problem."""

import math
from npc import Npc, NpcMemory, build_prompt


class FakeEngine:
    """Deterministic stand-in. No model needed to test the logic."""

    def embed_text(self, text):
        v = [0.0] * 16
        for ch in text.lower():
            v[ord(ch) % 16] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def generate(self, prompt, **kw):
        return f"[reply to {len(prompt)} chars]"


class BruteForceIndex:
    """What an HNSW index should agree with. Exact, slow, obviously correct."""

    def __init__(self, dim):
        self.items = []

    def insert(self, id_, vec):
        self.items.append((id_, vec))

    def query(self, vec, k):
        scored = [(sum(a * b for a, b in zip(vec, v)), i) for i, v in self.items]
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]


def make_npc(name="Mara", persona="You are Mara, a blacksmith."):
    eng = FakeEngine()
    npc = Npc(name, persona, eng)
    npc.memory = NpcMemory(name, 16, BruteForceIndex(16))
    return npc


fails = 0


def check(cond, what):
    global fails
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        fails += 1


print("prompt assembly")
p = build_prompt("You are Mara.", [], "Hello")
check("<|im_start|>system" in p, "has a system block")
check(p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
      "ends ready for the model to continue, thinking pre-empted")
check("remember" not in p, "no memory section when nothing was recalled")

print("\nmemory recall")
npc = make_npc()
npc.say("My brother Tomas went missing near the old mill")
npc.say("What do you charge for a horseshoe")
_, recalled = npc.say("Have you heard anything about my brother")
text = " ".join(r.text for r in recalled)
check("brother" in text or "Tomas" in text,
      "recalls the brother turn despite an off-topic turn in between")

print("\nmemory is per-npc")
a, b = make_npc("Mara"), make_npc("Bram", "You are Bram, an innkeeper.")
a.say("The sword is cursed")
check(len(b.memory) == 0, "one npc's memories don't leak into another's")
check(len(a.memory) == 2, "one exchange stores two records (player + npc)")

print("\n" + ("all good" if fails == 0 else "SOMETHING FAILED"))
raise SystemExit(0 if fails == 0 else 1)
