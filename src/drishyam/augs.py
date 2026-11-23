"""Anti-shortcut augmentations — train split only. choice order shuffled every step; score never."""

import random


def shuffled_options(options: list[str], gold: list[float], primitive: str, rng: random.Random):
    """Returns (options', gold') — permutation applied consistently (spec §3: choice only)."""
    if primitive != "choice":
        return options, gold  # ordinal: NEVER shuffled
    idx = list(range(len(options)))
    rng.shuffle(idx)
    return [options[i] for i in idx], [gold[i] for i in idx]


def jitter_state(state: str, rng: random.Random, p_case=0.2, p_ws=0.2) -> str:
    """20% of steps: uppercase 15-30% of words and/or append trailing newlines (spec §3)."""
    out = state
    if rng.random() < p_case:
        words = out.split(" ")
        words = [w.upper() if rng.random() < rng.uniform(0.15, 0.30) else w for w in words]
        out = " ".join(words)
    if rng.random() < p_ws:
        out += "\n" * rng.randint(1, 3)
    return out
