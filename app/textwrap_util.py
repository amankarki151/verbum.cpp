"""Word-wrap by measured pixel width, not character count -- character-count
wrapping breaks badly with a proportional font, where 'i' and 'w' aren't the
same width."""


def wrap_text(text, max_width, measure_fn):
    """measure_fn(str) -> pixel width, e.g. font.size(s)[0] in Pygame.
    Greedy wrap: keep adding words until the next one would overflow."""
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for w in words[1:]:
        trial = current + " " + w
        if measure_fn(trial) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines