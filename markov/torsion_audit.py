from __future__ import annotations
import argparse, json, sys, h5py
from pathlib import Path
from sage.all import GF, ZZ, Integer, PolynomialRing, HyperellipticCurve, factor, sqrt, is_prime

"""torsion_audit.py

Check 5: Torsion audit for the DLP contradiction diagnosis.

Hypothesis: some atoms (raw x-coordinates in Fp) correspond to divisor
classes D in J(C) that satisfy ell*D = 0 but D != 0, i.e. they are
genuine ell-torsion.  When such a class appears in a walk relation with
coefficient c, the linearised log-space row asserts c*log(D) = something,
but log(D) is only defined mod ord(D) which divides ell — so the relation
is globally consistent but locally wrong in GF(ell).  This produces exactly
the symptom: walk structure consistent with known key, but anchor causes
contradiction.

What this test does
-------------------
For every atom x0 in the factor base (i.e. every column of the relation
matrix), we:

  1. Lift x0 to a Mumford-form degree-1 or degree-2 reduced divisor on C.
     An x-coordinate x0 in Fp yields (at most) two y-values from C:
         y^2 = f(x0)   where f = x^5 + 3x^3 + 2x^2 + 5x + 4
     If f(x0) = 0  → ramification point → degree-1 divisor [x0, 0] - [∞]
     If f(x0) = □  → two points         → degree-1 divisor [x0, y] - [∞]
                                           (we take the positive square root)
     Either way the atom as used in the walk is a *degree-1 class*.

  2. Compute D = ell * (divisor class) in J(C) using Cantor's algorithm.
     If D == 0 (the identity) and the class is non-trivial → ell-torsion.

  3. Also compute the actual order of D by trial division of ell*cofactor
     (or just checking ell*D, (ell/q)*D for small prime factors of ell if
     ell is not prime — but ell=25373 is prime so ell*D is the only check
     that matters).

Since ell is prime the only ell-torsion in J[ell](Fp) would be classes
killed by ell.  For a *generic* genus-2 Jacobian over Fp the ell-torsion
is trivial in the p-order subgroup (that's the whole point of index calculus
— you work in the ell-order subgroup).  So finding ell*D = 0 for any
non-trivial D would be very surprising and would confirm the hypothesis.

The more likely culprit is a COFACTOR issue: #J(C)(Fp) = h * ell where h
is the cofactor.  If a divisor D has order h (or divides h) rather than
ell, then D is NOT in the ell-subgroup, and any relation involving D is
wrong in the ell-log space.  This is the "effective torsion" scenario.

So we also check: h * D == 0 (torsion in the cofactor subgroup).

Usage
-----
    sage -python torsion_audit.py relation_matrix.h5 \\
        --p 16411 --ell 25373 \\
        [--curve "x^5 + 3*x^3 + 2*x^2 + 5*x + 4"] \\
        [--full-order N]   # #J(C)(Fp) = N if known; else we use p*ell as upper bound

The --curve argument is optional: it defaults to x^5 + 3*x^3 + 2*x^2 + 5*x + 4.

Output
------
For every atom:
  TORSION-ELL    : ell * D == 0  (genuine ell-torsion — extremely suspicious)
  TORSION-COF    : cofactor * D == 0, ell * D != 0  (cofactor subgroup)
  OUTSIDE-SUBGRP : neither killed by ell nor cofactor — order doesn't divide full_order
  OK             : ell * D != 0  (divisor is in the right subgroup)
  NOSQRT         : f(x0) is a non-square in Fp (no point exists — bad atom)
  ZERO           : trivial divisor class (impossible for a factor-base atom)

Summary at the end: counts per category, and which atoms are enriched in
bad vs good rows.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)

def _section(title: str) -> None:
    sep = "=" * 70
    _log(f"\n{sep}")
    _log(f"  {title}")
    _log(sep)

def load_atoms_and_bad_good(hdf5_path: str):
    """Return (atoms, bad_atom_set, good_atom_set) from the HDF5 file.

    'atoms' is the full column list (strings).
    We don't rerun the contradiction filter here — we read the atom lists
    from the matrix structure and let the caller supply bad/good sets, OR
    we return all atoms if no incremental filter has been stored.

    Since the HDF5 doesn't cache check-4 results, we just return the full
    atom list and leave enrichment to the caller.
    """
    with h5py.File(hdf5_path, "r") as f:
        atoms = [a.decode("utf-8") for a in f["atoms"][:]]
    return atoms

def _parse_curve_coeffs(expr: str, Fp):
    """Parse a polynomial string like 'x^5 + 3*x^3 + ...' into a list of
    Fp coefficients [c0, c1, ..., c5] for c0 + c1*x + ... + c5*x^5."""
    R = PolynomialRing(Fp, 'x')
    try:
        poly = R(expr)
    except Exception as e:
        sys.exit(f"ERROR: cannot parse curve polynomial '{expr}': {e}")
    if poly.degree() != 5:
        sys.exit(f"ERROR: curve polynomial must be degree 5, got {poly.degree()}")
    return poly

def classify_divisor(x0_int: int, Fp, curve, ell: int, cofactor: int | None):
    """
    Classify the divisor class for atom x0_int.

    Returns a tuple (status, order_info) where status is one of:
        'ell-torsion'   ell * D = 0  (D is ell-torsion — bad)
        'cof-torsion'   cofactor * D = 0, ell * D != 0  (cofactor subgroup — bad)
        'ok'            neither — D lives in the prime subgroup  (good)
        'nosqrt'        f(x0) not a square  (bad atom regardless)
        'zero'          trivial divisor  (shouldn't happen)

    order_info is a dict with diagnostic info.
    """
    lift_status, D = _lift_to_divisor(x0_int, Fp, curve)
    if lift_status != 'ok':
        return lift_status, {}

    J = curve.jacobian()
    zero = J.zero()

    ell_D = Integer(ell) * D
    info = {'ell*D == 0': ell_D == zero}

    if ell_D == zero:
        info['verdict'] = 'ell-torsion'
        return 'ell-torsion', info

    if cofactor is not None and cofactor > 1:
        cof_D = Integer(cofactor) * D
        info['cof*D == 0'] = cof_D == zero
        if cof_D == zero:
            info['verdict'] = 'cof-torsion'
            return 'cof-torsion', info

    info['verdict'] = 'ok'
    return 'ok', info

# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def torsion_audit(
    hdf5_path: str,
    p: int,
    ell: int,
    curve_str: str,
    full_order: int | None,
    bad_atoms: set[str] | None,
    good_atoms: set[str] | None,
    sample_size: int = 0,   # 0 = audit all atoms
):
    _section("CHECK 5: TORSION AUDIT")
    _log(f"  p={p}  ell={ell}  curve: y^2 = {curve_str}")

    Fp = GF(Integer(p))
    R  = PolynomialRing(Fp, 'x')
    f  = R(curve_str)
    C  = HyperellipticCurve(f)

    cofactor = None
    if full_order is not None:
        if full_order % ell != 0:
            _log(f"  ⚠  full_order={full_order} is not divisible by ell={ell}!")
            _log("     Proceeding but cofactor check may be wrong.")
        else:
            cofactor = full_order // ell
            _log(f"  #J(C)(Fp) = {full_order} = {cofactor} * {ell}  (cofactor={cofactor})")
    else:
        _log("  --full-order not supplied.  Only ell-torsion check will run.")
        _log("  (to check cofactor subgroup, rerun with --full-order N)")

    atoms = load_atoms_and_bad_good(hdf5_path)

    # Filter to non-special atoms (skip ∞ and any non-integer atoms)
    fb_atoms = []
    for a in atoms:
        if a == "∞":
            continue
        try:
            int(a)
            fb_atoms.append(a)
        except ValueError:
            continue

    if sample_size > 0 and sample_size < len(fb_atoms):
        import random
        random.seed(7)
        fb_atoms = random.sample(fb_atoms, sample_size)
        _log(f"  Sampling {sample_size} of {len(atoms)} atoms for speed.")
    else:
        _log(f"  Auditing all {len(fb_atoms)} factor-base atoms ...")

    counts = {
        'ell-torsion': [],
        'cof-torsion': [],
        'ok':          [],
        'nosqrt':      [],
        'zero':        [],
    }

    bad_results  = {}   # atom -> status
    good_results = {}

    for atom_str in fb_atoms:
        x0 = int(atom_str)
        status, info = classify_divisor(x0, Fp, C, ell, cofactor)
        counts[status].append(atom_str)

        if bad_atoms and atom_str in bad_atoms:
            bad_results[atom_str] = status
        if good_atoms and atom_str in good_atoms:
            good_results[atom_str] = status

    # --- Summary ---
    _log("\n  ── Results by category ──")
    for cat, members in counts.items():
        flag = "  *** PROBLEM" if cat in ('ell-torsion', 'cof-torsion', 'nosqrt') and members else ""
        _log(f"  {cat:15s}: {len(members):5d} atom(s){flag}")
        if cat in ('ell-torsion', 'cof-torsion') and members:
            preview = members[:20]
            _log("    atoms: " + ", ".join(preview) + ("  ..." if len(members) > 20 else ""))

    if counts['nosqrt']:
        _log(f"\n  ⚠  {len(counts['nosqrt'])} atoms have no rational point over Fp!")
        _log("     These x-coordinates don't correspond to any point on C.")
        _log("     Preview: " + ", ".join(counts['nosqrt'][:20]))

    # --- Bad-atom enrichment ---
    if bad_atoms:
        _log("\n  ── Bad-atom torsion profile ──")
        _log(f"  (atoms flagged as enriched in contradiction rows by Check 4)")
        if not bad_results:
            _log("  None of the supplied bad atoms appear in the factor base.")
        else:
            for atom_str, status in sorted(bad_results.items()):
                flag = "  ***" if status != 'ok' else ""
                _log(f"    atom {atom_str:8s}: {status}{flag}")

    if good_atoms:
        _log("\n  ── Control: good-atom torsion profile ──")
        if not good_results:
            _log("  None of the supplied good atoms appear in the factor base.")
        else:
            for atom_str, status in sorted(good_results.items()):
                _log(f"    atom {atom_str:8s}: {status}")

    # --- Verdict ---
    _log("\n  ── Verdict ──")
    n_bad_torsion = len(counts['ell-torsion']) + len(counts['cof-torsion'])
    n_nosqrt      = len(counts['nosqrt'])

    if counts['ell-torsion']:
        _log("  ✗  ELL-TORSION ATOMS FOUND.")
        _log("     These divisors are killed by ell in J(C)(Fp).")
        _log("     Any relation involving them is wrong in GF(ell) log-space.")
        _log("     Fix: filter these x-coordinates out of the factor base entirely.")
    elif counts['cof-torsion']:
        _log("  ✗  COFACTOR-SUBGROUP ATOMS FOUND.")
        _log("     These divisors live in the cofactor part of J(C)(Fp), not the")
        _log("     ell-order subgroup.  Log-space is only defined mod ell for the")
        _log("     ell-subgroup; cofactor atoms produce garbage relations.")
        _log("     Fix: project every walk step onto the ell-subgroup by multiplying")
        _log("          by cofactor before Cantor reduction, OR filter these atoms.")
    elif n_nosqrt:
        _log("  ✗  ATOMS WITH NO RATIONAL POINT.")
        _log("     These x-coords have no lift to C(Fp).  They should never appear")
        _log("     in a factor base over Fp.")
        _log("     Fix: check your point-lifting / Cantor-coefficient extraction code.")
    elif n_bad_torsion == 0 and n_nosqrt == 0:
        _log("  ✓  All audited atoms are in the ell-subgroup.  Torsion is NOT the cause.")
        _log("     The contradiction must come from a different source.")
        _log("     Next steps: inspect the 9 bad rows directly (recompute from walk state).")

    return counts

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    DEFAULT_CURVE = "x^5 + 3*x^3 + 2*x^2 + 5*x + 4"

    ap = argparse.ArgumentParser(
        description="Torsion audit: check whether factor-base atoms lie in the ell-subgroup of J(C)."
    )
    ap.add_argument("hdf5_path", help="Path to HDF5 relation matrix")
    ap.add_argument("--p",   type=int, required=True,  help="Field characteristic")
    ap.add_argument("--ell", type=int, required=True,  help="Prime subgroup order")
    ap.add_argument("--curve", default=DEFAULT_CURVE,
                    help=f"Curve polynomial f s.t. y^2=f(x)  (default: {DEFAULT_CURVE})")
    ap.add_argument("--full-order", type=int, default=None,
                    help="#J(C)(Fp) — needed for cofactor-torsion check")
    ap.add_argument("--sample", type=int, default=0,
                    help="Audit only this many randomly sampled atoms (0 = all)")
    # Bad/good atoms from check 4: supplied as comma-separated lists
    ap.add_argument("--bad-atoms", default=None,
                    help="Comma-separated atom list from Check 4 bad rows (for enrichment report)")
    ap.add_argument("--good-atoms", default=None,
                    help="Comma-separated atom list from Check 4 good rows (for control)")
    args = ap.parse_args(argv)

    if not Path(args.hdf5_path).exists():
        sys.exit(f"ERROR: file not found: {args.hdf5_path}")

    bad_atoms  = set(args.bad_atoms.split(","))  if args.bad_atoms  else None
    good_atoms = set(args.good_atoms.split(",")) if args.good_atoms else None

    # Hardcoded enriched-bad-atoms from the Check 4 output as a convenience:
    # 3095, 14919, 11156, 11944, 6584, 879, 4090, 5474, 4088, 16372
    if bad_atoms is None:
        _log("  [hint] No --bad-atoms supplied.  Using enriched atoms from Check 4 output:")
        _log("         3095,14919,11156,11944,6584,879,4090,5474,4088,16372")
        bad_atoms = {"3095","14919","11156","11944","6584","879","4090","5474","4088","16372"}

    torsion_audit(
        hdf5_path  = args.hdf5_path,
        p          = args.p,
        ell        = args.ell,
        curve_str  = args.curve,
        full_order = args.full_order,
        bad_atoms  = bad_atoms,
        good_atoms = good_atoms,
        sample_size= args.sample,
    )

    _log(f"\n{'#'*70}")
    _log("# TORSION AUDIT COMPLETE")
    _log(f"{'#'*70}\n")

def _lift_to_divisor(x0_int: int, Fp, curve):
    J = curve.jacobian()
    f = curve.hyperelliptic_polynomials()[0]
    x0 = Fp(x0_int)
    rhs = f(x0)

    if rhs == Fp(0):
        P = curve(x0, Fp(0))
        D = J(P)
        return ('ok', D)

    # Use extend=False so Sage raises ArithmeticError instead of going to GF(p^2)
    try:
        y0 = rhs.sqrt(extend=False)
    except (ArithmeticError, ValueError):
        return ('nosqrt', None)

    P = curve(x0, y0)
    D = J(P)
    if D == J.zero():
        return ('zero', J.zero())
    return ('ok', D)

if __name__ == "__main__":
    main()
