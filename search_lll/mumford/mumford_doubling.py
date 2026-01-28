from sage.all import QQ, ZZ, GF, PolynomialRing, HyperellipticCurve
from collections import defaultdict
from .mumford_core import _poly_from_coeffs_qq, make_monic, reduce_v_mod_u, is_divisor_on_curve
from ..rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError

#from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError

def compute_doubled_point_modular(D_start, f_coeffs, num_doublings, primes_list, debug=False):
    """
    Calculates 2^num_doublings * D_start using parallel modular arithmetic
    and Rational Reconstruction (CRT).
    D_start is a SageMath Jacobian element over QQ.
    """
    R_QQ = D_start[0].parent()
    u_coeffs_current, v_coeffs_current = _get_divisor_coeffs_qq(D_start)

    # Determine max expected degree for u and v based on genus g=2
    # u degree is g=2 (length 3), v degree is g-1=1 (length 2)
    u_max_len = 3
    v_max_len = 2

    for n in range(num_doublings):
        u_coeffs_next = defaultdict(list)
        v_coeffs_next = defaultdict(list)

        if debug:
            print(f"  [MOD-DBL] Doubling iteration {n+1}/{num_doublings}")

        success_count = 0
        for p in primes_list:
            if p == 2:
                continue

            # Prepare coefficients reduced mod p
            try:
                # Need the *integer* numerator/denominator for a safe mod reduction
                # We catch ZeroDivisionError specifically for p dividing the denominator
                u_mod_p = [int(c.numerator()) * pow(int(c.denominator()), -1, p) % p for c in u_coeffs_current]
                v_mod_p = [int(c.numerator()) * pow(int(c.denominator()), -1, p) % p for c in v_coeffs_current]
            except (ZeroDivisionError, ValueError):
                # Denominator is divisible by p, skip this prime
                raise
                continue

            u_2p_coeffs, v_2p_coeffs = _mumford_doubling_mod_p_internal(
                u_mod_p, v_mod_p, f_coeffs, p
            )

            # Validate output from _mumford_doubling_mod_p_internal and skip bad primes
            if u_2p_coeffs is None or v_2p_coeffs is None:
                # modular routine signalled this prime is unusable; skip it
                if debug:
                    print(f"  [MOD-DBL] prime {p} produced no valid doubled divisor (skipping).")
                continue

            # Ensure output lists are integer residues in 0..p-1
            try:
                u_2p_coeffs = [int(x) % p for x in u_2p_coeffs]
                v_2p_coeffs = [int(x) % p for x in v_2p_coeffs]
            except Exception:
                if debug:
                    print(f"  [MOD-DBL] prime {p} returned non-integer coefficients (skipping).")
                raise
                continue

            if u_2p_coeffs is None:
                continue

            success_count += 1

            # Pad modular coeffs with leading zeros to match max degree
            u_2p_coeffs = [0] * (u_max_len - len(u_2p_coeffs)) + u_2p_coeffs
            v_2p_coeffs = [0] * (v_max_len - len(v_2p_coeffs)) + v_2p_coeffs

            # Store results by coefficient index
            for i in range(u_max_len):
                u_coeffs_next[i].append((p, int(u_2p_coeffs[i])))
            for i in range(v_max_len):
                v_coeffs_next[i].append((p, int(v_2p_coeffs[i])))

        # --- Rational Reconstruction ---
        u_reconstructed = []
        v_reconstructed = []

        # Check if we have enough primes

        # --- Choose only primes that contributed to *every* coefficient (avoid Frankenstein mixes) ---
        # Build list of primes that provided values for all u and v coefficient positions
        good_primes = []
        for p in primes_list:
            if p == 2:
                continue
            ok = True
            for i in range(u_max_len):
                if not any(pp == p for pp, _ in u_coeffs_next.get(i, [])):
                    ok = False
                    break
            if not ok:
                continue
            for i in range(v_max_len):
                if not any(pp == p for pp, _ in v_coeffs_next.get(i, [])):
                    ok = False
                    break
            if ok:
                good_primes.append(p)

        if debug:
            print(f"  [MOD-DBL] good_primes (present for all coeffs) = {good_primes}")

        # require a minimum number of shared primes to reconstruct the whole polynomial safely
        if len(good_primes) < MIN_SUCCESS_PRIMES:
            if debug:
                print(f"  [MOD-DBL] Critical failure: only {len(good_primes)} fully-consistent primes available.")
            raise ValueError(f"Modular doubling failed at iteration {n+1} due to insufficient consistent primes ({len(good_primes)}).")

        # We will reconstruct every coefficient using the SAME good_primes (same modulus M)
        primes_for_all = tuple(good_primes)

        # helper to extract residues for a coefficient in the order of primes_for_all
        def coeff_residues_for_primes(coeff_list):
            # coeff_list is list of (p,val) for that coefficient
            lookup = {p: val for p, val in coeff_list}
            return tuple(lookup[p] for p in primes_for_all)

        try:

            # Reconstruct u coefficients (using same primes_for_all for each coeff)
            M_c = math.prod(primes_for_all)
            for i in range(u_max_len):
                if not u_coeffs_next[i]:
                    u_reconstructed.append(QQ(0))
                    continue

                vals_for_c = coeff_residues_for_primes(u_coeffs_next[i])
                crt_val = crt_cached(vals_for_c, primes_for_all)
                num, den = rational_reconstruct(crt_val, M_c)

                if abs(num) > M_c**RECON_EXPONENT or abs(den) > M_c**RECON_EXPONENT:
                    raise RationalReconstructionError(
                        f"Height too large for coeff {i}: num={num}, den={den}, M_c={M_c}, exponent={RECON_EXPONENT}"
                    )

                u_reconstructed.append(QQ(num)/QQ(den))

            # Reconstruct v coefficients (same primes_for_all)
            for i in range(v_max_len):
                if not v_coeffs_next[i]:
                    v_reconstructed.append(QQ(0))
                    continue

                vals_for_c = coeff_residues_for_primes(v_coeffs_next[i])
                crt_val = crt_cached(vals_for_c, primes_for_all)
                num, den = rational_reconstruct(crt_val, M_c)

                if abs(num) > M_c**RECON_EXPONENT or abs(den) > M_c**RECON_EXPONENT:
                    raise RationalReconstructionError(
                        f"Height too large for coeff {i}: num={num}, den={den}, M_c={M_c}, exponent={RECON_EXPONENT}"
                    )

                v_reconstructed.append(QQ(num)/QQ(den))

            # Form the new Mumford divisor 2*D_current over Q[x]
            u_next = _poly_from_coeffs_qq(R_QQ, u_reconstructed)
            v_next = _poly_from_coeffs_qq(R_QQ, v_reconstructed)

            # Recreate the SageMath Jacobian element
            f_poly_qq = _poly_from_coeffs_qq(R_QQ, f_coeffs)
            C_QQ = HyperellipticCurve(f_poly_qq, R_QQ(0))
            J_QQ = C_QQ.jacobian()

            # assume u_next, v_next, f_poly are present as Sage polynomials over QQ
            # try cheap repairs then validate
            u_try = make_monic(u_next)
            v_try = reduce_v_mod_u(v_next, u_try)

            valid, reason = is_divisor_on_curve(u_try, v_try, f_poly_qq)
            if not valid:
                # second attempt: sometimes denominator-scaling leaves remainder; try clearing denominators
                try:
                    # clear denominators of coefficients for both u_try and v_try
                    den_lcm = 1
                    for coeff in u_try.coefficients() + v_try.coefficients():
                        den_lcm = lcm(den_lcm, QQ(coeff).denominator())
                    # scale to integer polynomials (work over ZZ) then reduce v mod u again
                    u_int = (u_try * den_lcm).change_ring(ZZ)
                    v_int = (v_try * den_lcm).change_ring(ZZ)
                    # convert back to QQ and re-normalize to monic u
                    u_scaled = PolynomialRing(QQ, 'x')(u_int).change_ring(QQ)
                    v_scaled = PolynomialRing(QQ, 'x')(v_int).change_ring(QQ)
                    u_scaled = make_monic(u_scaled)
                    v_scaled = reduce_v_mod_u(v_scaled, u_scaled)
                    valid2, reason2 = is_divisor_on_curve(u_scaled, v_scaled, f_poly)
                    if valid2:
                        u_try, v_try = u_scaled, v_scaled
                        valid, reason = True, None
                except Exception:
                    # ignore integer-scaling failure, will re-raise below
                    raise

            if not valid:
                # give a very explicit error for upstream handling and logging
                msg = (f"Reconstructed (u,v) failed Mumford test after repair attempts: {reason}.\n"
                    f"u = {u_next}\n"
                    f"v = {v_next}\n"
                    f"u_try = {u_try}\n"
                    f"v_try = {v_try}\n")
                if debug:
                    print("[compute_doubled_point_modular] " + msg)
                # raise so caller treats this as a reconstruction failure (and can skip it)
                raise RationalReconstructionError(msg)

            # If valid, construct Jacobian point from repaired pair
            u_next, v_next = u_try, v_try
            u_next = make_monic(u_next)
            v_next = v_next % u_next
            if (v_next**2 - f_poly_qq) % u_next != 0:
                raise RationalReconstructionError("v^2 != f mod u after doubling")

            D_current = J_QQ([u_next, v_next])

        except RationalReconstructionError as e:
            if debug:
                print(f"  [MOD-DBL] modular reconstruction failed at doubling {n+1}: {e}")
                print("  [MOD-DBL] Falling back to exact QQ doubling for the remaining iterations (slower but exact).")

            # Fallback: reconstruct exact divisor from current rational coeffs and finish doublings exactly
            try:
                # current divisor (exact) from the integer/Q rational coeff lists
                u_exact = _poly_from_coeffs_qq(R_QQ, u_coeffs_current)
                v_exact = _poly_from_coeffs_qq(R_QQ, v_coeffs_current)
                u_exact = make_monic(u_exact)
                v_exact = reduce_v_mod_u(v_exact, u_exact)

                f_poly_qq = _poly_from_coeffs_qq(R_QQ, f_coeffs)
                C_QQ = HyperellipticCurve(f_poly_qq, R_QQ(0))
                J_QQ = C_QQ.jacobian()
                D_exact = J_QQ([u_exact, v_exact])

                # finish remaining doublings exactly
                remaining = num_doublings - n
                for _ in range(remaining):
                    D_exact = 2 * D_exact

                # return the exact result for the caller
                return D_exact

            except Exception as ee:
                # If exact fallback fails, re-raise the original modular error as ValueError
                raise ValueError(f"Modular doubling failed at iteration {n+1} and exact fallback failed too: {e}") from ee
            raise

    return D_current

