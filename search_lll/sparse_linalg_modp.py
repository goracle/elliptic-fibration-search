import random
from sage.all import Integer, Zmod
from sage.matrix.berlekamp_massey import berlekamp_massey
from multiprocessing import Pool, cpu_count

class SparseRelationMatrix:
    def __init__(self, rows, rhs, modulus):
        """
        rows: list of dicts {col: coeff}
        rhs:  list of ints
        modulus: prime ℓ
        """
        self.rows = rows
        self.rhs = rhs
        self.mod = modulus
        self.K = Zmod(modulus)

        self.n_rows = len(rows)
        self.n_cols = max(
            max(r.keys()) if r else 0 for r in rows
        ) + 1

        # build column-wise view for transpose matvec
        self.cols = [[] for _ in range(self.n_cols)]
        for i, r in enumerate(rows):
            for j, v in r.items():
                self.cols[j].append((i, v % modulus))


def _matvec_chunk(args):
    rows, vec, mod = args
    out = [0] * len(rows)
    for i, r in rows:
        s = 0
        for j, v in r.items():
            s += v * vec[j]
        out[i] = s % mod
    return out


def parallel_matvec(rows, vec, mod, nprocs=20):
    if nprocs is None:
        nprocs = max(1, cpu_count() - 1)

    chunks = [[] for _ in range(nprocs)]
    for i, r in enumerate(rows):
        chunks[i % nprocs].append((i, r))

    with Pool(nprocs) as pool:
        parts = pool.map(
            _matvec_chunk,
            [(chunk, vec, mod) for chunk in chunks]
        )

    out = [0] * len(rows)
    for part in parts:
        for i, v in enumerate(part):
            if v:
                out[i] = v
    return out


def block_wiedemann_solve(A, b, block_size=8, iters=None, verbose=True):
    """
    Solve A x = b mod ℓ using Block Wiedemann.
    A: SparseRelationMatrix
    b: RHS vector (length n_rows)
    """
    mod = A.mod
    K = Zmod(mod)

    n = A.n_cols
    m = A.n_rows

    if iters is None:
        iters = 2 * n // block_size + 20

    if verbose:
        print(f"[BW] block={block_size}, iters={iters}, cores={cpu_count()}")

    # random start vectors
    V = [[random.randrange(mod) for _ in range(n)]
         for _ in range(block_size)]

    # Krylov sequence
    seq = []
    for t in range(iters):
        if verbose and t % 10 == 0:
            print(f"[BW] iter {t}/{iters}")

        AV = []
        for v in V:
            Av = parallel_matvec(A.rows, v, mod)
            AV.append(Av)

        # compute projections with RHS
        for Av in AV:
            seq.append(sum((Av[i] * b[i]) % mod for i in range(m)) % mod)

        # lift: V ← Aᵀ A V
        V_new = []
        for Av in AV:
            vT = [0] * n
            for j, col in enumerate(A.cols):
                s = 0
                for i, c in col:
                    s += c * Av[i]
                vT[j] = s % mod
            V_new.append(vT)
        V = V_new

    # minimal polynomial
    poly = berlekamp_massey(seq)
    if verbose:
        print(f"[BW] minimal polynomial degree {poly.degree()}")

    # solve via polynomial back-substitution
    x = [0] * n
    for coeff, vec in zip(poly.list(), V):
        if coeff:
            for i in range(n):
                x[i] = (x[i] + coeff * vec[i]) % mod

    return vector(K, x)


def solve_dlp_mod_l_block_wiedemann(valid_rows, rhs, row_q, beta_q,
                                   full_order, G, Q, verbose=True):
    ℓ = int(get_largest_prime_factor(full_order))
    h = int(Integer(full_order) // ℓ)

    if verbose:
        print(f"[DLP] Solving mod ℓ={ℓ} using Block Wiedemann")

    A = SparseRelationMatrix(valid_rows, rhs, ℓ)
    b = [(v - beta_q) % ℓ for v in rhs]

    sol = block_wiedemann_solve(A, b, verbose=verbose)

    # compute discrete log
    d = 0
    for j, v in row_q.items():
        d = (d + v * sol[j]) % ℓ

    d = Integer(d)

    # verify in ℓ-subgroup
    if d * (h * G) != h * Q:
        raise RuntimeError("Block Wiedemann solution failed verification")

    if verbose:
        print("[DLP] ✓ discrete log verified in ℓ-subgroup")

    return d
