"""Runs an NPC's generate() call on a background thread so Pygame's event
loop keeps rendering while the model is thinking.

generate() has py::call_guard<py::gil_scoped_release>() on the C++ side
(Day 11), which releases the GIL for the duration of the call -- but that
only helps if something actually calls it from a different thread than the
one running the render loop. This is that something.
"""

import queue
import threading


class AsyncNpcCaller:
    """One request in flight at a time. A submission while busy is rejected,
    not queued -- so clicking an NPC repeatedly during a slow generation
    doesn't pile up requests."""

    def __init__(self):
        self.result_q = queue.Queue()
        self.busy = False

    def submit(self, npc, player_line, **kwargs):
        if self.busy:
            return False
        self.busy = True

        def worker():
            reply, recalled = npc.say(player_line, **kwargs)
            self.result_q.put((reply, recalled))

        threading.Thread(target=worker, daemon=True).start()
        return True

    def poll(self):
        """Call once per frame. Returns (reply, recalled) if a result
        arrived this frame, else None. Never blocks."""
        try:
            result = self.result_q.get_nowait()
            self.busy = False
            return result
        except queue.Empty:
            return None