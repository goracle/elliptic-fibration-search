"""
search_lll/known_ledger.py

Two small helpers, both reporting-only (they never steer the search):

  * x_to_m / KnownLedger: track what is known about the WHOLE curve (global x
    values) and translate to the current fibration's local m only when a
    per-step summary needs it.  m is local to a fibration (m = a*x + b, with
    b fixed by the seed point), so an m from a previous fibration is noise.

  * group_chains_by_m: after the per-vector graph runs, collect the m each
    confirmed chain reconstructs to and group equal m across different n.
    A point that shows up at several n is the cheap cross-n signal; no extra
    CRT work is done.
"""

from collections import defaultdict


def linear_m_map(r_m, shift):
    """
    Return (a, b) with m = a*x + b for the current fibration, where x is the
    global (unshifted) coordinate.  Uses the same convention as
    search_main._record_rational_candidate:  x = r_m(m) - shift.
    Returns None if r_m is not linear (then callers skip the translation).
    """
    from sage.all import QQ
    try:
        x0 = QQ(r_m(m=QQ(0))) - QQ(shift)
        x1 = QQ(r_m(m=QQ(1))) - QQ(shift)
        x2 = QQ(r_m(m=QQ(2))) - QQ(shift)
    except Exception:
        return None
    slope = x1 - x0            # dx/dm
    if slope == 0 or (x2 - x1) != slope:
        return None
    a = 1 / slope
    b = -x0 / slope
    return a, b


class KnownLedger:
    """Known x-values for the whole curve, viewed through one fibration."""

    def __init__(self, known_x, r_m, shift):
        self.known_x = set(known_x)
        self.map = linear_m_map(r_m, shift)
        self.found_by = {}     # x -> set of source labels ('graph', 'sweep')

    def m_of(self, x):
        if self.map is None:
            return None
        a, b = self.map
        return a * x + b

    def record(self, x, source):
        self.found_by.setdefault(x, set()).add(source)

    def summary_line(self, all_known_x):
        """One line: how many of the known x's each method has recovered."""
        total = len(all_known_x)
        g = sum(1 for x in all_known_x if 'graph' in self.found_by.get(x, ()))
        s = sum(1 for x in all_known_x if 'sweep' in self.found_by.get(x, ()))
        either = sum(1 for x in all_known_x if self.found_by.get(x))
        def _size(q):
            n, d = q.numerator, q.denominator
            n = n() if callable(n) else n
            d = d() if callable(d) else d
            return abs(n) + abs(d)
        missed = sorted((x for x in all_known_x if not self.found_by.get(x)), key=_size)
        line = (f"[known] {total} x known | this fibration recovered {either} "
                f"(graph {g}, sweep {s})")
        if missed:
            shown = ", ".join(str(x) for x in missed[:6])
            more = f" +{len(missed) - 6} more" if len(missed) > 6 else ""
            line += f" | not recovered: {shown}{more}"
        return line


def group_chains_by_m(per_vector_ms, min_vectors=2):
    """
    per_vector_ms: {v_tuple: iterable of (m_num, m_den)}.
    Returns {(m_num, m_den): sorted list of v_tuples that reconstructed it},
    keeping only m reconstructed at >= min_vectors distinct vectors.
    """
    seen = defaultdict(set)
    for v, ms in per_vector_ms.items():
        for m in set(ms):
            seen[m].add(v)
    return {m: sorted(vs) for m, vs in seen.items() if len(vs) >= min_vectors}
