from __future__ import annotations
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from sage.all import GF, ZZ, QQ, HyperellipticCurve, PolynomialRing

"""cantor_cache.py

Pair-level Cantor reduction cache for the genus-2 Markov walk.

Philosophy
----------
Each accepted walk relation looks like

    3·xi + xj + xk - 5·∞ = 0          (degree-5 curve)

The *three* unordered pairs from that relation —

    {xi, xj},  {xi, xk},  {xj, xk}

— correspond to degree-2 effective divisors.  Cantor reduction maps each such
divisor to its *unique* reduced representative in Jac(C).  If two pairs from
*different* relations reduce to the same representative, the two divisors are
linearly equivalent, which is a hidden collision that the x-coordinate walk
alone would miss.

We deliberately avoid random-subset explosion by only ever looking at pairs
(size-2 subsets).  Cross-relation triple combinations are not attempted unless
explicitly requested and are off by default.

What the cache gives you
------------------------
1.  **Early collision detection** — hidden Jacobian equivalences discovered as
    each new relation is added, before any x-coordinate repeats.

2.  **Atom substitution hints** — when a new pair reduces to a known
    representative, we surface which existing walk atoms can replace the new
    ones, giving the relation-matrix builder a handle for FB compression.

3.  **Cheap consistency check** — involution symmetry (xj ↔ xk) implies that
    {xi,xj} and {xi,xk} from the same relation may or may not reduce to the
    same class; checking catches algebra bugs in xk recovery.

Cost model
----------
Per new relation:  at most 3 Cantor-reduce calls (one per pair).
Per history replay: len(history) * 3 calls.
Cantor reduction for genus-2 over F_p is O(deg^2) = O(1), so this is
proportional to the number of relations — never exponential.

Usage
-----
    from cantor_cache import CantorPairCache

    cache = CantorPairCache(C, p)           # C = HyperellipticCurve or poly

    # Feed relations one at a time (call from your step loop):
    hits = cache.add_relation(xi, xj, xk)

    # Or replay a full walker history:
    hits = cache.replay_history(walker.history)

    cache.summary()

Each returned hit is a CantorHit namedtuple with fields:
    new_pair, existing_pair, reduced_rep, relation_index_new, relation_index_old
"""

# ---------------------------------------------------------------------------
# Sage imports — raise on failure so the caller sees it clearly
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReducedRep:
    """Cantor reduced representative of a degree-≤2 divisor on C.

    Stored as the pair (u, v) of polynomials in the Mumford representation
    D ~ div(u, v) with deg v < deg u and u | (v^2 - f).
    We use their string representations as hash keys so Sage element identity
    is not required.
    """
    u_str: str
    v_str: str

    @classmethod
    def from_mumford(cls, u, v) -> "ReducedRep":
        return cls(u_str=str(u), v_str=str(v))

    def __str__(self):
        return f"D({self.u_str}, {self.v_str})"

@dataclass
class CantorHit:
    """A discovered hidden collision between two walk relations."""
    # The pair that just arrived
    new_pair: FrozenSet
    # The pair already in the cache that it collided with
    existing_pair: FrozenSet
    # Their shared reduced representative
    reduced_rep: ReducedRep
    # Index of the relation that introduced new_pair  (0-based into history)
    relation_index_new: int
    # Index of the relation that introduced existing_pair
    relation_index_old: int
    # Which slot in the relation produced new_pair  ("xi-xj", "xi-xk", "xj-xk")
    new_slot: str
    # Which slot produced the existing pair
    old_slot: str

    def __str__(self):
        return (
            f"[CantorHit] rel#{self.relation_index_new}({self.new_slot}) "
            f"≡ rel#{self.relation_index_old}({self.old_slot})  "
            f"via {self.reduced_rep}"
        )

def _pair_slot(xi, xj, xk, pair: FrozenSet) -> str:
    """Label which atom-slot produced this pair."""
    a, b = tuple(pair)
    s = frozenset
    if pair == s([xi, xj]):
        return "xi-xj"
    if pair == s([xi, xk]):
        return "xi-xk"
    if pair == s([xj, xk]):
        return "xj-xk"
    return "unknown"

# ---------------------------------------------------------------------------
# Core cache
# ---------------------------------------------------------------------------

class CantorPairCache:
    """Cache of Cantor-reduced pair divisors accumulated over a walk.

    Parameters
    ----------
    curve_or_poly : HyperellipticCurve or polynomial f such that C: y^2 = f(x)
    p             : prime  (required; all arithmetic is over F_p)
    curve_degree  : degree of the hyperelliptic polynomial (default 5)
    check_involution : if True, warn when {xi,xj} and {xi,xk} from the same
                       relation do NOT reduce to different classes (degenerate step)
    """

    def __init__(
        self,
        curve_or_poly,
        p: int,
        *,
        curve_degree: int = 5,
        check_involution: bool = True,
        verbose: bool = True,
    ):
        self.p = int(p)
        self.Fp = GF(self.p)
        self.curve_degree = curve_degree
        self.check_involution = check_involution
        self.verbose = verbose

        # Build the HyperellipticCurve if we got a polynomial
        if isinstance(curve_or_poly, (list, tuple)):
            R = PolynomialRing(self.Fp, "x")
            x = R.gen()
            coeffs = list(curve_or_poly)
            deg = len(coeffs) - 1
            f = sum(self.Fp(coeffs[i]) * x ** (deg - i) for i in range(len(coeffs)))
            self.C = HyperellipticCurve(f)
        elif hasattr(curve_or_poly, "parent") and hasattr(curve_or_poly, "degree"):
            # It's already a polynomial
            R = curve_or_poly.parent()
            try:
                f = R.change_ring(self.Fp)(curve_or_poly)
            except Exception:
                f = curve_or_poly
            self.C = HyperellipticCurve(f)
        else:
            # Assume it's already a HyperellipticCurve
            self.C = curve_or_poly

        self.J = self.C.jacobian()

        # -- primary index: ReducedRep -> list of (pair, relation_index, slot) --
        self._rep_to_entries: Dict[ReducedRep, List[Tuple[FrozenSet, int, str]]] = defaultdict(list)

        # -- secondary index: pair -> (rep, relation_index, slot) for fast lookup --
        self._pair_to_entry: Dict[FrozenSet, Tuple[ReducedRep, int, str]] = {}

        self._n_relations: int = 0
        self._hits: List[CantorHit] = []
        self._n_cantor_calls: int = 0

        # atom substitution map: atom -> set of atoms it's been seen equivalent-with
        self._equiv_atoms: Dict[Any, set] = defaultdict(set)

    # ------------------------------------------------------------------
    # Cantor reduction
    # ------------------------------------------------------------------

    def _reduce_pair(self, xa, xb) -> Optional[ReducedRep]:
        """Cantor-reduce the degree-2 divisor {xa} + {xb} on C over F_p.

        Returns None if the pair is degenerate (xa == xb, or either is not on C).
        """
        if xa == xb:
            return None

        Fp = self.Fp
        J = self.J
        self._n_cantor_calls += 1

        try:
            xa_fp = Fp(xa)
            xb_fp = Fp(xb)
        except Exception as e:
            raise ValueError(f"Cannot coerce to F_p: {xa!r}, {xb!r}: {e}") from e

        try:
            # Lift to points on C
            Pa = self.C.lift_x(xa_fp)
            Pb = self.C.lift_x(xb_fp)
            # Form the divisor and reduce in J
            D = J(Pa) + J(Pb)
            u, v = D
            return ReducedRep.from_mumford(u, v)
        except Exception:
            # xa or xb not on C over F_p, or arithmetic failure — skip silently
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_relation(
        self,
        xi,
        xj,
        xk,
        relation_index: Optional[int] = None,
    ) -> List[CantorHit]:
        """Register one walk relation and return any new collision hits.

        Parameters
        ----------
        xi, xj, xk     : atom x-coordinates from the relation
        relation_index  : caller-supplied index (defaults to internal counter)

        Returns
        -------
        List of CantorHit (may be empty).
        """
        if relation_index is None:
            relation_index = self._n_relations
        self._n_relations += 1

        new_hits: List[CantorHit] = []

        # The three geometrically meaningful pairs from this relation
        pairs_with_slots: List[Tuple[FrozenSet, str]] = []
        s = frozenset
        if xj is not None:
            pairs_with_slots.append((s([xi, xj]), "xi-xj"))
        if xk is not None:
            pairs_with_slots.append((s([xi, xk]), "xi-xk"))
        if xj is not None and xk is not None:
            pairs_with_slots.append((s([xj, xk]), "xj-xk"))

        # Optional: warn if {xi,xj} and {xi,xk} are already equivalent (degenerate)
        if self.check_involution and xj is not None and xk is not None:
            r_ij = self._reduce_pair(xi, xj)
            r_ik = self._reduce_pair(xi, xk)
            if r_ij is not None and r_ik is not None and r_ij == r_ik:
                if self.verbose:
                    print(
                        f"[CantorCache] WARNING rel#{relation_index}: "
                        f"{{xi,xj}} ≡ {{xi,xk}} in Jac — degenerate step?"
                    )

        for pair, slot in pairs_with_slots:
            if len(pair) < 2:
                continue  # skip if xa == xb

            # Skip if we've already cached this exact pair
            if pair in self._pair_to_entry:
                continue

            xa, xb = tuple(pair)
            rep = self._reduce_pair(xa, xb)
            if rep is None:
                continue

            # Check for collision with previously seen pairs
            existing = self._rep_to_entries.get(rep, [])
            for (old_pair, old_rel_idx, old_slot) in existing:
                hit = CantorHit(
                    new_pair=pair,
                    existing_pair=old_pair,
                    reduced_rep=rep,
                    relation_index_new=relation_index,
                    relation_index_old=old_rel_idx,
                    new_slot=slot,
                    old_slot=old_slot,
                )
                new_hits.append(hit)
                self._hits.append(hit)
                self._record_equiv_atoms(pair, old_pair)
                if self.verbose:
                    print(f"  {hit}")

            # Store in both indexes
            self._rep_to_entries[rep].append((pair, relation_index, slot))
            self._pair_to_entry[pair] = (rep, relation_index, slot)

        return new_hits

    def replay_history(self, history, *, accepted_only: bool = True) -> List[CantorHit]:
        """Replay a full walker history list and accumulate all hits.

        Handles both RelationRecord dataclasses and plain dicts.
        """
        all_hits: List[CantorHit] = []

        def _get(rec, key):
            if isinstance(rec, dict):
                return rec.get(key)
            return getattr(rec, key, None)

        for idx, rec in enumerate(history):
            if accepted_only and not _get(rec, "accepted"):
                continue

            # Skip involution-closure synthetic records — they add no new pairs
            step = _get(rec, "step")
            if isinstance(step, dict) and step.get("source") == "involution_closure":
                continue

            xi = _get(rec, "xi")
            xj = _get(rec, "xj")
            xk = _get(rec, "xk")

            if xi is None:
                continue

            hits = self.add_relation(xi, xj, xk, relation_index=idx)
            all_hits.extend(hits)

            # Also sweep xj/xk candidates from the pool if present
            pool = _get(rec, "candidate_pool") or []
            for cand in pool:
                if not isinstance(cand, dict):
                    continue
                c_xj = cand.get("xj") or cand.get("x") or cand.get("candidate_x")
                c_xk = cand.get("xk")
                if c_xj is not None and c_xj != xj:
                    hits = self.add_relation(xi, c_xj, c_xk, relation_index=idx)
                    all_hits.extend(hits)

        return all_hits

    # ------------------------------------------------------------------
    # Atom equivalence map (for FB substitution hints)
    # ------------------------------------------------------------------

    def _record_equiv_atoms(self, pair_new: FrozenSet, pair_old: FrozenSet):
        """Track which atoms have been identified as equivalent via a hit."""
        for a in pair_new:
            for b in pair_old:
                if a != b:
                    self._equiv_atoms[a].add(b)
                    self._equiv_atoms[b].add(a)

    def equiv_partners(self, atom) -> set:
        """Return the set of atoms that have been seen as Jacobian-equivalent to atom."""
        return set(self._equiv_atoms.get(atom, set()))

    def can_substitute(self, atom) -> bool:
        """True if atom has at least one known Jacobian-equivalent partner."""
        return bool(self._equiv_atoms.get(atom))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def lookup_pair(self, xa, xb) -> Optional[ReducedRep]:
        """Return the cached reduced rep for this pair, or None if unseen."""
        pair = frozenset([xa, xb])
        entry = self._pair_to_entry.get(pair)
        return entry[0] if entry else None

    def hits_for_atom(self, atom) -> List[CantorHit]:
        """Return all hits that involved a given atom."""
        return [h for h in self._hits if atom in h.new_pair or atom in h.existing_pair]

    def n_unique_reps(self) -> int:
        """Number of distinct reduced representatives seen so far."""
        return len(self._rep_to_entries)

    def n_pairs_cached(self) -> int:
        return len(self._pair_to_entry)

    def all_hits(self) -> List[CantorHit]:
        return list(self._hits)

    # ------------------------------------------------------------------
    # Integration hook: call this from your step loop
    # ------------------------------------------------------------------

    def on_new_step(self, rec) -> List[CantorHit]:
        """Convenience wrapper to call from inside a walker step loop.

        Pass the RelationRecord (or dict) directly.  Returns any hits.
        """
        def _get(r, k):
            if isinstance(r, dict):
                return r.get(k)
            return getattr(r, k, None)

        if not _get(rec, "accepted"):
            return []
        step = _get(rec, "step")
        if isinstance(step, dict) and step.get("source") == "involution_closure":
            return []

        return self.add_relation(
            _get(rec, "xi"),
            _get(rec, "xj"),
            _get(rec, "xk"),
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self):
        n_hits = len(self._hits)
        n_reps = self.n_unique_reps()
        n_pairs = self.n_pairs_cached()

        print("\n" + "=" * 65)
        print("CANTOR PAIR CACHE SUMMARY")
        print("=" * 65)
        print(f"  Relations processed  : {self._n_relations}")
        print(f"  Pairs cached         : {n_pairs}")
        print(f"  Unique reduced reps  : {n_reps}")
        print(f"  Cantor reduce calls  : {self._n_cantor_calls}")
        print(f"  Hidden collisions    : {n_hits}")
        print(f"  Atoms with equiv     : {len(self._equiv_atoms)}")

        if n_hits:
            print(f"\n  Collision list:")
            for i, h in enumerate(self._hits):
                print(f"    [{i}] {h}")

        if self._equiv_atoms:
            print(f"\n  Atom equivalence map (substitution hints):")
            for atom, partners in sorted(
                self._equiv_atoms.items(), key=lambda kv: -len(kv[1])
            )[:20]:
                print(f"    {atom}  ≡  {sorted(partners, key=str)[:5]}")
            if len(self._equiv_atoms) > 20:
                print(f"    ... ({len(self._equiv_atoms)} atoms total)")

        print("=" * 65 + "\n")

# ---------------------------------------------------------------------------
# Convenience: attach to walker class
# ---------------------------------------------------------------------------

def attach_cantor_cache_to_walker(walker_class, curve_or_poly=None, p=None, **kwargs):
    """Monkey-patch a CantorPairCache onto a walker class.

    After calling this, the walker instance will have:
        walker.cantor_cache        — the live CantorPairCache
        walker.cantor_summary()    — print the cache summary

    The cache is lazily initialised on the first call to walker.step()
    because the curve poly and p are available on the instance.

    Alternatively pass curve_or_poly and p to pre-build it.

    Example
    -------
        from cantor_cache import attach_cantor_cache_to_walker
        attach_cantor_cache_to_walker(Genus2MetropolisWalker)

        walker = build_default_walker(...)
        walker.run(200)
        walker.cantor_summary()
        hits = walker.cantor_cache.all_hits()
    """
    import types

    original_step = walker_class.step

    def _ensure_cache(self):
        if not hasattr(self, "cantor_cache") or self.cantor_cache is None:
            cp = curve_or_poly if curve_or_poly is not None else self.curve_poly
            pp = p if p is not None else self.p
            if pp is None:
                raise ValueError(
                    "CantorPairCache requires a prime p.  "
                    "Pass p= to attach_cantor_cache_to_walker or set walker.p."
                )
            self.cantor_cache = CantorPairCache(cp, pp, **kwargs)

    def step_with_cache(self, n=None, seed=None):
        _ensure_cache(self)
        rec = original_step(self, n=n, seed=seed)
        if rec is not None:
            hits = self.cantor_cache.on_new_step(rec)
            if hits and getattr(self.config, "verbose", True):
                for h in hits:
                    print(f"  [cantor_cache] {h}")
        return rec

    def cantor_summary(self):
        _ensure_cache(self)
        self.cantor_cache.summary()

    walker_class.step = step_with_cache
    walker_class.cantor_summary = cantor_summary
    # Initialise to None so the lazy check works
    walker_class.cantor_cache = None
