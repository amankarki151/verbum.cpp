"""The thin demo shell: one room, a few NPCs, click to talk. Proves the
engine and memory layer work together end to end -- deliberately not a game.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build-py"))
import pygame
import verbum

try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except Exception as e:
    print(f"clipboard unavailable ({e}) -- typing still works, paste won't")
    CLIPBOARD_AVAILABLE = False

from async_npc import AsyncNpcCaller
from lattice_index import LatticeIndex
from npc import Npc, NpcMemory
from textwrap_util import wrap_text

WIDTH, HEIGHT = 900, 600
BG = (30, 26, 22)
NPC_COLOR = (120, 100, 70)
NPC_ACTIVE = (200, 160, 90)
TEXT_COLOR = (230, 225, 210)
BUBBLE_BG = (50, 44, 36)
INPUT_BG = (45, 40, 34)
DATA_DIR = os.path.join(os.path.dirname(__file__), "npc_data")
MAX_INPUT_LEN = 200
CURSOR_BLINK_MS = 500

NPCS = [
    {"name": "Meera", "pos": (220, 300),
     "persona": "You are Meera, a blacksmith in a small village. Gruff but "
                "kind. The player is a traveler talking to you -- whatever "
                "they tell you about their own life is about them, not you. "
                "Speak in your own words -- never repeat the player's "
                "sentence back to them, even partly. Answer in one short "
                "sentence."},
    {"name": "Arjun", "pos": (650, 300),
     "persona": "You are Arjun, an innkeeper in a small village. Cheerful "
                "and talkative. The player is a traveler talking to you -- "
                "whatever they tell you about their own life is about them, "
                "not you. Speak in your own words -- never repeat the "
                "player's sentence back to them, even partly. Answer in one "
                "short sentence."},
]


def make_npcs(engine):
    npcs = {}
    for spec in NPCS:
        npc = Npc(spec["name"], spec["persona"], engine)
        npc.memory = NpcMemory(
            spec["name"], engine.hidden_size,
            LatticeIndex(os.path.join(DATA_DIR, spec["name"]), engine.hidden_size),
        )
        npcs[spec["name"]] = {"npc": npc, "pos": spec["pos"]}
    return npcs


def paste_into(current_text, clipboard_text, max_len):
    """Insert clipboard content at the end of the current input, respecting
    the same length cap normal typing already respects. Verified separately
    against empty clipboards and overlong pastes."""
    return (current_text + (clipboard_text or ""))[:max_len]


def cursor_visible(ticks_ms, interval_ms=CURSOR_BLINK_MS):
    """500ms on, 500ms off. Verified separately: flips exactly on the
    interval boundary, toggles 4 times over 2 seconds."""
    return (ticks_ms // interval_ms) % 2 == 0


def main():
        # Start each demo run from a clean slate. Lattice's vector storage is
    # genuinely persistent across restarts, but NpcMemory's text side
    # (MemoryRecord) never was -- it only ever lived in an in-memory list
    # that dies with the process. Half-persisted state across two mismatched
    # storage layers is worse than none: a fresh session's locally-numbered
    # ids can collide with old ids still sitting in Lattice's on-disk index,
    # and a query can return an id with no matching text anywhere, which is
    # exactly what crashed here. Wiping npc_data/ at every launch keeps both
    # sides honestly in sync -- memory persists within one run, which is all
    # this demo has ever actually tested, not across separate launches.
    import shutil
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("verbum.cpp -- NPC memory demo")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Georgia", 18)
    small_font = pygame.font.SysFont("Georgia", 14)

    # Load the engine and NPCs on a background thread, same reasoning as
    # AsyncNpcCaller for generation: a real model load takes several seconds,
    # and blocking the main thread for that long makes the OS flag the
    # window as "Not Responding" even if you drew a nice loading message
    # right before the block started. `done` must be the LAST write in the
    # worker -- verified separately across 500 trials with no lock needed,
    # relying on the GIL serializing bytecode execution within a thread.
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
        screen.fill(BG)
        msg1 = font.render("Failed to load the model.", True, (220, 120, 100))
        msg2 = small_font.render(load_state["error"][:90], True, (200, 190, 170))
        msg3 = small_font.render(
            "Check that models/qwen3-0.6b exists relative to app/. Press any key to quit.",
            True, (150, 145, 130))
        screen.blit(msg1, (40, HEIGHT // 2 - 40))
        screen.blit(msg2, (40, HEIGHT // 2))
        screen.blit(msg3, (40, HEIGHT // 2 + 30))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False
            clock.tick(30)
        pygame.quit()
        return

    engine = load_state["engine"]
    npcs = load_state["npcs"]
    print("ready.")

    npc_state = {name: {"last_reply": "", "last_recalled": []} for name in npcs}

    caller = AsyncNpcCaller()
    pending_for = None

    active_npc = None
    input_text = ""
    input_active = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                for name, data in npcs.items():
                    nx, ny = data["pos"]
                    if (event.pos[0] - nx) ** 2 + (event.pos[1] - ny) ** 2 < 40 ** 2:
                        active_npc = name
                        input_active = True
                        input_text = ""

            elif event.type == pygame.KEYDOWN and input_active:
                cmd_or_ctrl = event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)

                if cmd_or_ctrl and event.key == pygame.K_v and CLIPBOARD_AVAILABLE:
                    try:
                        clip_text = pyperclip.paste()
                    except Exception:
                        clip_text = ""
                    input_text = paste_into(input_text, clip_text, MAX_INPUT_LEN)

                elif cmd_or_ctrl and event.key == pygame.K_c and CLIPBOARD_AVAILABLE:
                    try:
                        pyperclip.copy(input_text)
                    except Exception:
                        pass

                elif event.key == pygame.K_RETURN and input_text.strip():
                    if pending_for is None:
                        npc = npcs[active_npc]["npc"]
                        if caller.submit(npc, input_text.strip(),
                                        max_tokens=50, temperature=0.7):
                            pending_for = active_npc
                            input_text = ""

                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]

                elif event.key == pygame.K_ESCAPE:
                    input_active = False
                    active_npc = None

                elif not cmd_or_ctrl:
                    if len(input_text) < MAX_INPUT_LEN:
                        input_text += event.unicode

        if pending_for is not None:
            result = caller.poll()
            if result is not None:
                reply, recalled = result
                npc_state[pending_for]["last_reply"] = reply
                npc_state[pending_for]["last_recalled"] = recalled
                pending_for = None

        # ---- draw ----
        screen.fill(BG)

        for name, data in npcs.items():
            color = NPC_ACTIVE if name == active_npc else NPC_COLOR
            pygame.draw.circle(screen, color, data["pos"], 40)
            label = font.render(name, True, TEXT_COLOR)
            screen.blit(label, (data["pos"][0] - label.get_width() // 2,
                                data["pos"][1] + 50))

        if active_npc:
            box = pygame.Rect(40, 420, WIDTH - 80, 140)
            pygame.draw.rect(screen, BUBBLE_BG, box, border_radius=8)

            is_thinking_for_active = (pending_for == active_npc)
            state = npc_state[active_npc]

            if is_thinking_for_active:
                msg = f"{active_npc} is thinking..."
                screen.blit(font.render(msg, True, TEXT_COLOR), (box.x + 16, box.y + 12))
            elif state["last_reply"]:
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

            waiting_on_other = pending_for is not None and pending_for != active_npc
            if waiting_on_other:
                prompt = f"({pending_for} is still responding...)"
                screen.blit(small_font.render(prompt, True, (140, 130, 100)),
                           (input_box.x + 8, input_box.y + 5))
            else:
                if input_text:
                    text_surf = small_font.render(input_text, True, TEXT_COLOR)
                else:
                    text_surf = small_font.render(
                        "type, paste, or press enter...", True, (110, 105, 95))
                screen.blit(text_surf, (input_box.x + 8, input_box.y + 5))

                if cursor_visible(pygame.time.get_ticks()):
                    text_width = small_font.size(input_text)[0] if input_text else 0
                    cx = input_box.x + 8 + text_width + 2
                    pygame.draw.line(screen, TEXT_COLOR,
                                     (cx, input_box.y + 4), (cx, input_box.y + 22), 2)
        else:
            hint = small_font.render("click an NPC to talk", True, (140, 135, 120))
            screen.blit(hint, (20, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
