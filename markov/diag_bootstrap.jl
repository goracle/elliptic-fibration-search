# diag_bootstrap.jl
# Package bootstrap and all `using` imports for the dlp_contradiction_diag suite.
# Include this first (and only once) in every entry point / driver script.
#!/usr/bin/env julia
"""
dlp_contradiction_diag.jl

Post-mortem diagnostics for a failed DLP solve, operating directly on the
HDF5 relation-matrix dump produced by dlp_diagnostics.dump_matrix_hdf5().

Four core checks:
  1. HOMOGENEOUS CHECK – does the known log-G vector lie in ker(A_hom)?
  2. CONTRADICTION CERTIFICATE – Farkas left-kernel witness for inconsistency.
  3. STRUCTURAL COLLAPSE TRIAGE – column fusion, special-col order, rank stability.
  4. INCREMENTAL CONSISTENCY FILTER – step-order Gaussian elimination.

Usage
-----
    julia dlp_contradiction_diag.jl relation_matrix.h5 \\
          --group-order 25373 --known-key 802

    # or, if group_order / known_key are stored in the HDF5 metadata:
    julia dlp_contradiction_diag.jl relation_matrix.h5
"""

import Pkg

# ---------------------------------------------------------------------------
# Dependency bootstrap — install missing packages silently on first run.
# ---------------------------------------------------------------------------
const REQUIRED_PKGS = ["HDF5", "JSON3", "ArgParse", "Nemo", "SparseArrays",
                       "LinearAlgebra"]

for pkg in REQUIRED_PKGS
    if !haskey(Pkg.project().dependencies, pkg) &&
       !any(p.name == pkg for p in values(Pkg.dependencies()))
        @info "Installing $pkg …"
        Pkg.add(pkg)
    end
end

using HDF5
using JSON3
using ArgParse
using Nemo          # GF, matrix, kernel, rank
using SparseArrays
using LinearAlgebra
using Random: seed!, randperm, MersenneTwister
using Printf  # Add this line
