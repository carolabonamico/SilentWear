"""
CTC decoding strategies, shared by the classification (lexicon) and
recognition (free-character, WER/CER) paths.

Two decoders operate on the pre-collapse per-frame log-probabilities:

- Greedy (best-path): argmax per frame gives an alignment path (blanks and
  repeats included); the CTC collapse (merge repeats, drop blanks) is then
  applied as the final step.
- Prefix beam search: standard label-synchronous CTC prefix beam search, no 
  external language model. The collapse rule is built into the search: 
  the beam holds already-collapsed prefixes and sums probability over all 
  alignments that collapse to the same prefix, so it can recover strings that 
  greedy misses.

The alphabet here is tiny (a-z + space + blank = 28), so an exact expansion over
all symbols per beam is cheap and no symbol pruning is needed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import math


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_prefix_beam_search(
    log_probs: Sequence[Sequence[float]],
    beam_width: int,
    blank_id: int = 0,
) -> List[int]:
    """Prefix beam search over a single utterance.

    Args:
        log_probs: (T, C) log-softmax probabilities (list/ndarray-like).
        beam_width: number of prefixes kept after each frame.
        blank_id: CTC blank index.

    Returns:
        The most probable collapsed (blank/repeat-removed) token-id sequence.
    """
    T = len(log_probs)
    if T == 0:
        return []
    C = len(log_probs[0])

    NEG = -math.inf
    # prefix (tuple of token ids) -> [p_blank, p_nonblank] in log space.
    # p_blank: prob of prefix with the last alignment step being a blank.
    # p_nonblank: prob of prefix with the last alignment step being a real symbol.
    beams = {(): [0.0, NEG]}

    for t in range(T):
        row = log_probs[t]
        next_beams: dict = {}

        def _get(prefix):
            e = next_beams.get(prefix)
            if e is None:
                e = [NEG, NEG]
                next_beams[prefix] = e
            return e

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _logsumexp(p_b, p_nb)
            last = prefix[-1] if prefix else -1

            for c in range(C):
                lp = row[c]
                if lp == NEG:
                    continue

                if c == blank_id:
                    # Staying on the same prefix via a blank frame.
                    e = _get(prefix)
                    e[0] = _logsumexp(e[0], p_total + lp)
                    continue

                if c == last:
                    # Repeat of the last emitted symbol: extending the prefix
                    # requires a separating blank, so it can only come from the
                    # blank-ending mass; the non-blank-ending mass collapses back
                    # onto the same prefix (a merged repeat).
                    e_new = _get(prefix + (c,))
                    e_new[1] = _logsumexp(e_new[1], p_b + lp)
                    e_same = _get(prefix)
                    e_same[1] = _logsumexp(e_same[1], p_nb + lp)
                else:
                    e_new = _get(prefix + (c,))
                    e_new[1] = _logsumexp(e_new[1], p_total + lp)

        # Keep the beam_width most probable prefixes.
        scored = sorted(
            next_beams.items(),
            key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            reverse=True,
        )
        beams = dict(scored[: max(1, beam_width)])

    best_prefix = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return list(best_prefix)


def ctc_greedy_decode(argmax_ids: Sequence[int], blank_id: int = 0) -> List[int]:
    """Best-path decode: collapse repeats then remove blanks from an argmax path."""
    collapsed: List[int] = []
    prev: Optional[int] = None
    for tok in argmax_ids:
        if tok == blank_id:
            prev = tok
            continue
        if tok != prev:
            collapsed.append(int(tok))
        prev = tok
    return collapsed
