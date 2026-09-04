"""End-to-end test: real verbum.cpp engine, real Lattice index. Same
off-topic-recall check as test_npc.py's fake version -- this is the proof
that swapping the fake index for the real one didn't change behavior."""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build-py"))
import verbum

from lattice_index import LatticeIndex
from npc import Npc, NpcMemory

DATA_DIR = "/tmp/verbum_npc_memory_test"
shutil.rmtree(DATA_DIR, ignore_errors=True)

print("loading engine...")
engine = verbum.Engine("../models/qwen3-0.6b")


def make_npc(name, persona):
    npc = Npc(name, persona, engine)
    npc.memory = NpcMemory(
        name, engine.hidden_size, LatticeIndex(f"{DATA_DIR}/{name}", engine.hidden_size)
    )
    return npc


fails = 0


def check(cond, what):
    global fails
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        fails += 1


npc = make_npc("Mara", "You are Mara, a blacksmith. Answer in one short sentence.")

npc.say("My brother Tomas went missing near the old mill", max_tokens=20)
npc.say("What do you charge for a horseshoe", max_tokens=20)
reply, recalled = npc.say("Have you heard anything about my brother", max_tokens=20)

text = " ".join(r.text for r in recalled)
check("brother" in text or "Tomas" in text,
      "real engine + real lattice: recalls the brother turn")
print("npc replied:", repr(reply))

print("\n" + ("all good" if fails == 0 else "SOMETHING FAILED"))
raise SystemExit(0 if fails == 0 else 1)