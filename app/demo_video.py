"""Auto-plays the exact demo sequence for recording — no manual clicking or
typing needed. Reuses every real piece of demo.py (personas, async
generation, rendering) so what's on screen is the genuine app, just driven
by a script instead of a mouse and keyboard.

Run it, hit record, don't touch anything until it says DONE at the bottom.
"""

import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build-py"))
import pygame
import verbum

from async_npc import AsyncNpcCaller
from demo import (
    WIDTH, HEIGHT, BG, NPC_COLOR, NPC_ACTIVE, TEXT_COLOR, BUBBLE_BG,
    INPUT_BG, NPCS, DATA_DIR, make_npcs, cursor_visible,
)
from textwrap_util import wrap_text

# ---- tune the pacing here if you want it faster/slower ----
COUNTDOWN_S = 3.0
TYPE_MS_PER_CHAR = 45
SUBMIT_DELAY_MS = 400
READ_PAUSE_S = 3.0        # how long every reply stays on screen once it appears
FINAL_EXTRA_WAIT_S = 3.0  # extra wait AFTER the last reply's hold, before DONE
FONT_NAME = "Georgia"

SCRIPT = [
    {"npc": "Meera", "message": "My brother Tomas went missing near the old mill."},
    {"npc": "Arjun", "message": "What's the best room here?"},
    {"npc": "Meera", "message": "Have you heard anything about my brother?"},
]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("verbum.cpp -- NPC memory demo")

    # Same fix as demo.py: Lattice's vectors persist on disk across runs,
    # but the text side never did -- wiping this at every launch keeps both
    # sides honestly in sync instead of colliding on stale ids.
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, 18)
    small_font = pygame.font.SysFont(FONT_NAME, 14)
    big_font = pygame.font.SysFont(FONT_NAME, 48)

    load_state = {"engine": None, "npcs": None, "error": None, "done": False}

    def load_worker():
        try:
            engine = verbum.Engine("../models/qwen3-0.6b")
            npcs = make_npcs(engine)
            load_state["engine"] = engine
            load_state["npcs"] = npcs
        except Exception as e:
            load_state["error"] = str(e)
        load_state["done"] = True

    print("loading engine (this takes a moment)...")
    threading.Thread(target=load_worker, daemon=True).start()

    dots = 0
    while not load_state["done"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        screen.fill(BG)
        dots = (dots + 1) % 90
        msg = "Loading engine" + "." * (1 + dots // 30)
        text = font.render(msg, True, TEXT_COLOR)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2,
                          HEIGHT // 2 - text.get_height() // 2))
        pygame.display.flip()
        clock.tick(30)

    if load_state["error"] is not None:
        print(f"failed to load: {load_state['error']}")
        pygame.quit()
        return

    npcs = load_state["npcs"]
    print("ready. starting scripted sequence...")

    npc_state = {name: {"last_reply": "", "last_recalled": []} for name in npcs}
    caller = AsyncNpcCaller()
    pending_for = None
    active_npc = None
    input_text = ""

    step_idx = 0
    phase = "countdown"
    phase_start = time.time()
    last_char_time = 0.0
    typed_chars = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time()

        if phase == "countdown":
            if now - phase_start >= COUNTDOWN_S:
                phase = "clicking"

        elif phase == "clicking":
            active_npc = SCRIPT[step_idx]["npc"]
            input_text = ""
            typed_chars = 0
            last_char_time = now
            phase = "typing"

        elif phase == "typing":
            message = SCRIPT[step_idx]["message"]
            if (now - last_char_time) * 1000 >= TYPE_MS_PER_CHAR:
                typed_chars += 1
                input_text = message[:typed_chars]
                last_char_time = now
                if typed_chars >= len(message):
                    phase = "about_to_submit"
                    phase_start = now

        elif phase == "about_to_submit":
            if (now - phase_start) * 1000 >= SUBMIT_DELAY_MS:
                npc = npcs[active_npc]["npc"]
                caller.submit(npc, input_text.strip(), max_tokens=50, temperature=0.7)
                pending_for = active_npc
                input_text = ""
                phase = "waiting_reply"

        elif phase == "waiting_reply":
            if pending_for is None:
                phase = "pausing"
                phase_start = now

        elif phase == "pausing":
            # Holds the just-arrived reply on screen for READ_PAUSE_S,
            # regardless of which step this is -- active_npc is already
            # correct here (set at the top of "clicking"), so the draw
            # logic below needs no step-index comparison at all.
            if (now - phase_start) >= READ_PAUSE_S:
                step_idx += 1
                if step_idx >= len(SCRIPT):
                    phase = "final_wait"
                    phase_start = now
                else:
                    phase = "clicking"

        elif phase == "final_wait":
            # Extra wait AFTER the last reply's own hold, with the reply
            # still fully visible, before DONE appears -- kept as a
            # separate phase so it never overlaps the reply's own display.
            if (now - phase_start) >= FINAL_EXTRA_WAIT_S:
                phase = "done"

        if pending_for is not None:
            result = caller.poll()
            if result is not None:
                reply, recalled = result
                npc_state[pending_for]["last_reply"] = reply
                npc_state[pending_for]["last_recalled"] = recalled
                pending_for = None

        # ---- draw ----
        screen.fill(BG)

        if phase == "countdown":
            remaining = max(0, int(COUNTDOWN_S - (now - phase_start)) + 1)
            text = big_font.render(str(remaining), True, TEXT_COLOR)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2,
                              HEIGHT // 2 - text.get_height() // 2))
        else:
            for name, data in npcs.items():
                color = NPC_ACTIVE if name == active_npc else NPC_COLOR
                pygame.draw.circle(screen, color, data["pos"], 40)
                label = font.render(name, True, TEXT_COLOR)
                screen.blit(label, (data["pos"][0] - label.get_width() // 2,
                                    data["pos"][1] + 50))

            if active_npc:
                box = pygame.Rect(40, 420, WIDTH - 80, 140)
                pygame.draw.rect(screen, BUBBLE_BG, box, border_radius=8)

                if pending_for == active_npc:
                    msg = f"{active_npc} is thinking..."
                    screen.blit(font.render(msg, True, TEXT_COLOR), (box.x + 16, box.y + 12))
                elif npc_state[active_npc]["last_reply"] and phase in ("pausing", "final_wait", "done"):
                    state = npc_state[active_npc]
                    lines = wrap_text(state["last_reply"], box.width - 32,
                                      lambda s: font.size(s)[0])
                    for i, line in enumerate(lines[:3]):
                        screen.blit(font.render(line, True, TEXT_COLOR),
                                   (box.x + 16, box.y + 12 + i * 24))
                    if state["last_recalled"]:
                        recall_str = "remembers: " + "; ".join(
                            r.text[:40] for r in state["last_recalled"])
                        screen.blit(small_font.render(recall_str, True, (150, 140, 110)),
                                   (box.x + 16, box.y + 100))

                input_box = pygame.Rect(40, 570, WIDTH - 80, 26)
                pygame.draw.rect(screen, INPUT_BG, input_box, border_radius=4)
                if input_text:
                    screen.blit(small_font.render(input_text, True, TEXT_COLOR),
                               (input_box.x + 8, input_box.y + 5))
                    if phase == "typing" and cursor_visible(pygame.time.get_ticks()):
                        w = small_font.size(input_text)[0]
                        cx = input_box.x + 8 + w + 2
                        pygame.draw.line(screen, TEXT_COLOR,
                                         (cx, input_box.y + 4), (cx, input_box.y + 22), 2)

            if phase == "done":
                done_text = small_font.render(
                    "DONE -- stop recording whenever you're ready", True, (150, 200, 150))
                screen.blit(done_text, (20, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()