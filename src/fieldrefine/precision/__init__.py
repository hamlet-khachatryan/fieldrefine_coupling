"""Restraint as precision: a separate formalism for the same inverse problem.

This is **not** an addition to the generator-based refinement in :mod:`core`. That
estimator pushes a latent ``z`` through operators and bolts restraints onto the objective
as penalties. This one makes the field itself the variable and the restraint *is* the
prior::

    p(drho) ~ exp(-0.5 drho' P drho),    P = P_stat + mu * L_W

There is no ``z``, no ``u``, no generator. GMRF_SPEC.md is the contract; :mod:`core` is
reused only as a library of verified primitives and is never modified by anything here.

Modules:

1. :mod:`fieldrefine.precision.gmrf` -- features, random features, the chunked ``Z Z^T``,
   the kNN alternative, and matrix-free Lanczos.
2. :mod:`fieldrefine.precision.estimate` -- the extended context, ``P_stat`` / ``P_graph``
   / ``P_apply``, the preconditioned NCG solver, the readouts, and the memory report.

Two invariants worth stating where they cannot be missed:

- **The null is one condition.** delta_R_free is zero exactly when P is a Fourier
  multiplier. Stationary spectra, isotropic smoothing and displacement-only graphs are all
  multipliers and all provably transfer nothing; everything the estimator can contribute
  is off-diagonal mass in P.
- **Z is never materialised.** At N ~ 1e6 and m = 128 a stored Z is ~1 GiB and at 512^3 it
  is ~130 GB. It is generated in row chunks inside the matvec, from the stored ``phi``.

Run from the repository root, where ``core`` lives::

    PYTHONPATH=src python -c "from fieldrefine.precision import estimate"
"""

__all__ = ["gmrf", "estimate"]
