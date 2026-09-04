"""NPC personas, memory, and prompt assembly.

The engine (C++) generates text and produces embeddings. Lattice (C++) stores
and searches the vectors. This module is the glue: who each NPC is, what they
remember, and how that gets turned into a prompt.
"""

from dataclasses import dataclass, field


@dataclass
class MemoryRecord:
    id: int
    speaker: str      # "player" or the NPC's name
    text: str
    turn: int


class NpcMemory:
    """One NPC's memories. Vectors live in the index; text lives here.

    A vector index stores vectors and ids, not strings -- so the text needs a
    side table keyed by the same ids.
    """

    def __init__(self, name, dim, index):
        self.name = name
        self.dim = dim
        self.index = index
        self.records = []

    def add(self, speaker, text, vec):
        rid = len(self.records)
        self.records.append(MemoryRecord(rid, speaker, text, rid))
        self.index.insert(rid, vec)
        return rid

    def recall(self, vec, k=3):
        if not self.records:
            return []
        return [self.records[i] for i in self.index.query(vec, k)]

    def __len__(self):
        return len(self.records)


def build_prompt(persona, recalled, player_line):
    """Assemble the prompt the model actually sees.

    Recalled memories go *before* the current line, inside the system block,
    so the model reads them as established context rather than as part of
    what the player just said.

    The empty <think></think> right after the assistant turn starts is not
    decorative -- Qwen3 is a hybrid reasoning model that emits a reasoning
    block by default. Without this, the model spends its token budget
    thinking out loud instead of speaking in character, and a short
    max_tokens cuts it off mid-thought before it ever answers. An empty,
    pre-closed think block tells the model the reasoning phase is already
    done, so it goes straight to the actual reply.
    """
    parts = [f"<|im_start|>system\n{persona}"]
    if recalled:
        lines = "\n".join(f"- {r.speaker}: {r.text}" for r in recalled)
        parts.append(f"\n\nThings you remember from earlier:\n{lines}")
    parts.append("<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{player_line}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


@dataclass
class Npc:
    name: str
    persona: str
    engine: object
    memory: NpcMemory = field(default=None)

    def say(self, player_line, k=2, max_tokens=60, temperature=0.8, seed=0):
        """One conversational turn: recall, generate, remember.

        Returns (reply, recalled) so the caller can show what the NPC
        remembered -- which is the whole point of the demo, and shouldn't be
        invisible.
        """
        vec = self.engine.embed_text(player_line)
        recalled = self.memory.recall(vec, k)

        prompt = build_prompt(self.persona, recalled, player_line)
        reply = self.engine.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, seed=seed
        )
        reply = reply.strip()

        self.memory.add("player", player_line, vec)
        self.memory.add(self.name, reply, self.engine.embed_text(reply))

        return reply, recalled
