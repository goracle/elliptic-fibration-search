"""
archimedean_optim.py: Logic for archimedean height minimization.
"""
from .search_config import math, Integer, QQ, TMAX, DEBUG
from sage.all import QQ

def archimedean_height_QQ(x):
    """
    A simple archimedean height for a QQ element x = a/b in lowest terms.
    Returns log(max(|a|, |b|, 1)).
    """
    try:
        a = Integer(x.numerator())
        b = Integer(x.denominator())
    except (AttributeError, TypeError) as e:
        raise TypeError("archimedean_height_QQ expects a rational (QQ) input") from e

    val = max(abs(int(a)), abs(int(b)), 1)
    return math.log(val)

def archimedean_height_of_integer(n):
    """A proxy for the archimedean height of an integer n."""
    return float(math.log(max(abs(int(n)), 1)))

def minimize_archimedean_t(m0, M, r_m_func, shift, tmax, max_steps=150, patience=6):
    """
    Given residue class m = m0 (mod M), search over m = m0 + t*M to find integer t that minimizes
    archimedean height of x = r_m(m) - shift.
    """
    best = []

    def eval_for_t(t):
        m_candidate = m0 + t * M
        if abs(t) > tmax:
            return None
        try:
            x_val = r_m_func(m=QQ(m_candidate)) - shift
            if not (hasattr(x_val, 'numerator') and hasattr(x_val, 'denominator')):
                return None
            score = archimedean_height_QQ(x_val)
            return (QQ(m_candidate), float(score))
        except (ZeroDivisionError, TypeError, ArithmeticError):
            if DEBUG: print("we're here, for some reason")
            return None

    center = eval_for_t(0)
    if center is not None:
        best.append(center)

    steps = 0
    no_improve = 0
    current_best_score = best[0][1] if best else float('inf')
    t = 1
    while steps < max_steps and no_improve < patience:
        for s in (t, -t):
            res = eval_for_t(s)
            steps += 1
            if res is None:
                continue
            m_cand, score = res
            best.append((m_cand, score))
            if score + 1e-12 < current_best_score:
                current_best_score = score
                no_improve = 0
            else:
                no_improve += 1
            if abs(s) >= tmax:
                no_improve = patience
                break
        t += 1

    # Deduplicate and sort
    unique = {}
    for m_cand, score in best:
        num = int(m_cand.numerator())
        den = int(m_cand.denominator())
        key = (num, den)
        if key not in unique or score < unique[key]:
            unique[key] = score

    sorted_candidates = sorted(((QQ(num) / QQ(den), sc) for (num, den), sc in unique.items()),
                               key=lambda z: z[1])

    return sorted_candidates[:3]

def minimize_archimedean_t_linear_const(m0, M, r_m_func, shift, tmax):
    """
    For linear r_m(m), find t minimizing archimedean height of x = r_m(m) - shift.
    """
    const_C = r_m_func(m=QQ(0))
    target = - (m0 + const_C + shift) / float(M)
    target = float(target)

    try:
        cand_t = set([math.floor(target), math.ceil(target), int(round(target))])
    except Exception:
        print("target =", target)
        print(type(target))
        raise

    # Clamp to allowed range
    cand_t = {max(-tmax, min(tmax, t)) for t in cand_t}

    results = []
    for t in sorted(cand_t):
        m_try = int(m0) + int(t) * int(M)
        x = -m_try - const_C - shift
        score = float(math.log(max(abs(x), 1)))
        results.append((t, m_try, int(x), score))

    # sort by score then by |x|
    results.sort(key=lambda z: (z[3], abs(z[2])))
    return results
