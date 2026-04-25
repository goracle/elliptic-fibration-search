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

    mult(x_src)*x_src + x_step + x_res + ... - deg*∞ = 0

where multiplicities come from the intersection polynomial roots.  The *three*
named unordered pairs from the principal named slots —

    {x_src, x_step},  {x_src, x_res},  {x_step, x_res}

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

3.  **Cheap consistency check** — involution symmetry (x_step ↔ x_res) implies
    that {x_src,x_step} and {x_src,x_res} from the same relation may or may not
    reduce to the same class; checking catches algebra bugs in x_res recovery.

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
    hits = cache.add_relation(x_src, x_step, x_res)

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
    # Which slot in the relation produced new_pair  ("src-step", "src-res", "step-res")
    new_slot: str
    # Which slot produced the existing pair
    old_slot: str

    def __str__(self):
        return (
            f"[CantorHit] rel#{self.relation_index_new}({self.new_slot}) "
            f"≡ rel#{self.relation_index_old}({self.old_slot})  "
            f"via {self.reduced_rep}"
        )

def _pair_slot(x_src, x_step, x_res, pair: FrozenSet) -> str:
    """Label which atom-slot produced this pair."""
    a, b = tuple(pair)
    s = frozenset
    if pair == s([x_src, x_step]):
        return "src-step"
    if pair == s([x_src, x_res]):
        return "src-res"
    if pair == s([x_step, x_res]):
        return "step-res"
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
    check_involution : if True, warn when {x_src,x_step} and {x_src,x_res} from the same
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
            Pa = self.C.lift_x(xa_fp)
        except (ValueError, TypeError):
            return None   # xa not on C over F_p
        try:
            Pb = self.C.lift_x(xb_fp)
        except (ValueError, TypeError):
            return None   # xb not on C over F_p
        # Jacobian arithmetic — let genuine errors propagate
        D = J(Pa) + J(Pb)
        u, v = D
        return ReducedRep.from_mumford(u, v)

    def reduce_triple(self, xa, xb, xc, fixed_xa, fixed_xb):
        """Find a consistent lift of the 5-atom principal divisor, projected to 4 atoms.

        Given five x-roots from a principal divisor (sum = 0 in J), where
        (fixed_xa, fixed_xb) are the two "known" atoms and (xa, xb, xc) are the
        triple to reduce, find any branch assignment for all five points such that

            J(fixed_xa, ±y) + J(fixed_xb, ±y) + J(xa, ±y) + J(xb, ±y) + J(xc, ±y) = 0

        then return the Mumford roots (r0, r1) of -(D_fixed) = D_triple.

        Returns (r0, r1) as GF(p) elements, or None if no consistent lift exists
        (e.g. one of the x-values is not on the curve over F_p).
        """
        C = self.C
        Fp = self.Fp
        J = self.J

        # Lift each x to both branches.  If x is not on C over F_p, bail out.
        # Sage does not support unary minus on hyperelliptic curve points directly,
        # so we construct the conjugate branch explicitly as C(x, -y).
        def _both_branches(x):
            try:
                P = C.lift_x(Fp(x))
            except (ValueError, TypeError):
                return None
            x_coord, y_coord = P[0], P[1]
            try:
                P_conj = C(x_coord, -y_coord)
            except Exception:
                return None
            return (P, P_conj)

        branches_fixed_a = _both_branches(fixed_xa)
        branches_fixed_b = _both_branches(fixed_xb)
        branches_a = _both_branches(xa)
        branches_b = _both_branches(xb)
        branches_c = _both_branches(xc)

        if any(b is None for b in (branches_fixed_a, branches_fixed_b,
                                    branches_a, branches_b, branches_c)):
            return None

        # Precompute all 8 D_triple values.
        import itertools
        triple_map = {}  # Mumford (u_str, v_str) -> (r0, r1)
        for Pa, Pb, Pc in itertools.product(branches_a, branches_b, branches_c):
            try:
                D = J(Pa) + J(Pb) + J(Pc)
                u, v = D
                key = (str(u), str(v))
                if key not in triple_map:
                    # Extract Mumford roots of D (the degree-2 u polynomial)
                    try:
                        rts = u.roots(multiplicities=False)
                    except Exception:
                        rts = []
                    if len(rts) == 2:
                        triple_map[key] = (rts[0], rts[1])
                    elif len(rts) == 1:
                        triple_map[key] = (rts[0], rts[0])
            except Exception:
                continue

        # Try all 4 fixed-pair lifts; check if -D_fixed is in triple_map.
        for Pfa, Pfb in itertools.product(branches_fixed_a, branches_fixed_b):
            try:
                D_fixed = J(Pfa) + J(Pfb)
                D_neg = -D_fixed
                u_neg, v_neg = D_neg
                key = (str(u_neg), str(v_neg))
                if key in triple_map:
                    return triple_map[key]
            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_relation(
        self,
        x_src,
        x_step,
        x_res,
        relation_index: Optional[int] = None,
    ) -> List[CantorHit]:
        """Register one walk relation and return any new collision hits.

        Parameters
        ----------
        x_src, x_step, x_res : atom x-coordinates from the relation
        relation_index        : caller-supplied index (defaults to internal counter)

        Returns
        -------
        List of CantorHit (may be empty).
        """
        if relation_index is None:
            relation_index = self._n_relations
        self._n_relations += 1

        new_hits: List[CantorHit] = []

        # The three geometrically meaningful pairs from the named slots
        pairs_with_slots: List[Tuple[FrozenSet, str]] = []
        s = frozenset
        if x_step is not None:
            pairs_with_slots.append((s([x_src, x_step]), "src-step"))
        if x_res is not None:
            pairs_with_slots.append((s([x_src, x_res]), "src-res"))
        if x_step is not None and x_res is not None:
            pairs_with_slots.append((s([x_step, x_res]), "step-res"))

        # Optional: warn if {x_src,x_step} and {x_src,x_res} are already equivalent (degenerate).
        # Compute and cache these reductions so the main loop below can reuse them.
        _precomputed: Dict[FrozenSet, Optional[ReducedRep]] = {}
        if x_step is not None:
            _precomputed[frozenset([x_src, x_step])] = self._reduce_pair(x_src, x_step)
        if x_res is not None:
            _precomputed[frozenset([x_src, x_res])] = self._reduce_pair(x_src, x_res)

        if self.check_involution and x_step is not None and x_res is not None:
            r_ss = _precomputed[frozenset([x_src, x_step])]
            r_sr = _precomputed[frozenset([x_src, x_res])]
            if r_ss is not None and r_sr is not None and r_ss == r_sr:
                if self.verbose:
                    print(
                        f"[CantorCache] WARNING rel#{relation_index}: "
                        f"{{x_src,x_step}} ≡ {{x_src,x_res}} in Jac — degenerate step?"
                    )

        for pair, slot in pairs_with_slots:
            if len(pair) < 2:
                continue  # skip if xa == xb

            # Skip if we've already cached this exact pair
            if pair in self._pair_to_entry:
                continue

            xa, xb = tuple(pair)
            # Reuse precomputed reduction if available, otherwise compute now
            rep = _precomputed.get(pair, self._reduce_pair(xa, xb))
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

        ``accepted_only`` gates whether the relation triple (x_src, x_step, x_res) itself
        is registered.  Candidate-pool entries are *always* swept because they
        are valid F_p points regardless of whether the step was accepted — they
        represent the full geometric neighbourhood explored at each step.
        """
        all_hits: List[CantorHit] = []

        def _get(rec, key):
            if isinstance(rec, dict):
                return rec.get(key)
            return getattr(rec, key, None)

        for idx, rec in enumerate(history):
            # Skip involution-closure synthetic records — they add no new pairs.
            step = _get(rec, "step")
            if isinstance(step, dict) and step.get("source") == "involution_closure":
                continue

            x_src  = _get(rec, "x_src")
            x_step = _get(rec, "x_step")
            x_res  = _get(rec, "x_res")

            # Register the relation triple only for accepted (or when not filtering).
            if x_src is not None and (not accepted_only or _get(rec, "accepted")):
                hits = self.add_relation(x_src, x_step, x_res, relation_index=idx)
                all_hits.extend(hits)

            # Always sweep the candidate pool — valid F_p points regardless of acceptance.
            if x_src is not None:
                pool = _get(rec, "candidate_pool") or []
                for cand in pool:
                    if not isinstance(cand, dict):
                        continue
                    c_x_step = cand.get("x_step") or cand.get("xj") or cand.get("x") or cand.get("candidate_x")
                    c_x_res  = cand.get("x_res") or cand.get("xk")
                    # Skip the already-registered accepted pair.
                    if c_x_step is not None and c_x_step == x_step and c_x_res == x_res:
                        continue
                    if c_x_step is not None:
                        hits = self.add_relation(x_src, c_x_step, c_x_res, relation_index=idx)
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

        Processes two sources of pairs:
        1. The accepted relation's (x_src, x_step, x_res) triple.
        2. All candidate-pool entries for this step — these are geometrically
           valid F_p points already computed by the walker, so sweeping them
           costs only Cantor-reduce calls (cheap) and surfaces hits much earlier
           than accepted-only feeding would.  Pool candidates are indexed as
           sub-entries of the same step_index so hit attribution is correct.
        """
        def _get(r, k):
            if isinstance(r, dict):
                return r.get(k)
            return getattr(r, k, None)

        step = _get(rec, "step")
        if isinstance(step, dict) and step.get("source") == "involution_closure":
            return []

        step_index = _get(rec, "step_index")
        x_src = _get(rec, "x_src")
        all_hits: List[CantorHit] = []

        # 1. Accepted relation triple.
        if _get(rec, "accepted") and x_src is not None:
            hits = self.add_relation(
                x_src,
                _get(rec, "x_step"),
                _get(rec, "x_res"),
                relation_index=step_index,
            )
            all_hits.extend(hits)

        # 2. Candidate pool — sweep regardless of accepted flag.
        # Each pool entry is a dict with at least one of x_step/xj/x/candidate_x and
        # optionally x_res/xk.  We use x_src from the record (the step's source node).
        if x_src is not None:
            pool = _get(rec, "candidate_pool") or []
            for cand in pool:
                if not isinstance(cand, dict):
                    continue
                c_x_step = cand.get("x_step") or cand.get("xj") or cand.get("x") or cand.get("candidate_x")
                c_x_res  = cand.get("x_res") or cand.get("xk")
                # Skip the already-accepted pair to avoid double-counting.
                if c_x_step is not None and c_x_step == _get(rec, "x_step") and c_x_res == _get(rec, "x_res"):
                    continue
                if c_x_step is not None:
                    hits = self.add_relation(x_src, c_x_step, c_x_res, relation_index=step_index)
                    all_hits.extend(hits)

        return all_hits

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
