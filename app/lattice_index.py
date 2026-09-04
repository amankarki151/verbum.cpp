"""Adapter wrapping Lattice's real API to the tiny interface NpcMemory
already expects: insert(id, vec) and query(vec, k) -> list of ids.

Lattice's Database is WAL-backed and persists to disk -- unlike the
brute-force stand-in from Day 11, an NPC's memory here survives between
separate runs of the demo. That's a real feature, not just an
implementation detail: talk to an NPC today, come back tomorrow, it still
remembers.
"""

import os

import lattice


class LatticeIndex:
    def __init__(self, path, dim):
        os.makedirs(path, exist_ok=True)
        self.db = lattice.Database(path)
        self.dim = dim

    def insert(self, id_, vec):
        self.db.insert(lattice.Vector(id_, vec))

    def query(self, vec, k):
        hits = self.db.search(vec, k=k)
        return [h.id for h in hits]