"""Stage 2: the adapter. A cached PDB entry becomes a ``core.Ctx`` on one fixed grid.

Conventions fixed here, verified against gemmi 0.7.5 and asserted by ``test_pipeline.py``:

* ``core.r2c`` is an orthonormal rfft in numpy's ``exp(-2pi i h.r)`` convention, while
  crystallographic structure factors use ``exp(+2pi i h.r)``.  Hence
  ``F_xtal = (V / sqrt(N)) * conj(core.r2c(rho))``.
* A symmetry operator ``x -> R x + t`` acts on a symmetric map's spectrum as
  ``F(h) = exp(2pi i (M h).t) * F(M h)`` with ``M = R^-T``.  On the rfft half grid
  ``F(M h)`` is either stored directly or is the conjugate of the stored ``F(-M h)``;
  the latter case sets ``sym_conj``.
* Everything crossing the boundary from gemmi (float32 grids) is cast to float64 once.
"""

import json
from pathlib import Path

import gemmi
import numpy as np

import core

from . import fetch

import jax.numpy as jnp

SAMPLE_RATE = 3.0
FREE_FRACTION_RANGE = (0.02, 0.15)
DERIVED_FIELDS = ("sym_idx", "sym_phase", "sym_conj")
DENSITY_CUTOFF = 1e-7
SOLVENT_K = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5)
SOLVENT_B = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
PROJECTOR_TOL = 1e-10
S_MIN = 0.1
RHO_FLOOR = 0.0
SMOOTH_FACTORS = (2, 3, 5, 7)
FLOAT64_DTYPES = (np.float64, np.complex128, np.int32, np.int64, np.bool_)


class PrepareError(RuntimeError):
    pass


def h_shape(shape):
    """Give the half-spectrum shape for a real grid.

    Args:
        shape: Real-space grid shape ``(n0, n1, n2)``.

    Returns:
        The rfft shape ``(n0, n1, n2 // 2 + 1)``.
    """
    return (shape[0], shape[1], shape[2] // 2 + 1)


def is_smooth(n):
    """Test whether an FFT length factors into small primes.

    Args:
        n: Candidate grid length.

    Returns:
        True if ``n`` factors entirely into 2, 3, 5 and 7, which is where the FFT is
        fast.
    """
    for factor in SMOOTH_FACTORS:
        while n % factor == 0:
            n //= factor
    return n == 1


def grid_shape(mtz, sample_rate=SAMPLE_RATE):
    """Choose the one grid shape used by every real-space array in the run.

    Sizes are raised until they are even, 2/3/5/7-smooth, and divisible by the space
    group's grid factors. The last condition makes every symmetry translation an
    exact grid vector, without which a symmetrised map cannot be represented.

    Args:
        mtz: Reflection file, for the resolution limit and the space group.
        sample_rate: Grid points per d_min.

    Returns:
        The grid shape as a tuple.
    """
    base = mtz.get_size_for_hkl(sample_rate=sample_rate)
    factors = mtz.spacegroup.operations().find_grid_factors()
    out = []
    for size, factor in zip(base, factors):
        n = int(size)
        while n % factor or n % 2 or not is_smooth(n):
            n += 1
        out.append(n)
    return tuple(out)


def signed_miller(shape):
    """Give the signed Miller index of every half-grid point.

    The first two axes run over positive and negative frequencies, the third over
    0..Nyquist only, which is what makes the array a half spectrum.

    Args:
        shape: Real-space grid shape.

    Returns:
        Three integer arrays of shape ``h_shape(shape)``, one per index.
    """
    n0, n1, n2 = shape
    H = h_shape(shape)
    a0, a1 = np.arange(n0), np.arange(n1)
    m0 = np.where(a0 <= n0 // 2, a0, a0 - n0)
    m1 = np.where(a1 <= n1 // 2, a1, a1 - n1)
    m2 = np.arange(H[2])
    return (
        np.broadcast_to(m0[:, None, None], H).astype(np.int64),
        np.broadcast_to(m1[None, :, None], H).astype(np.int64),
        np.broadcast_to(m2[None, None, :], H).astype(np.int64),
    )


def fold_indices(j0, j1, j2, shape):
    """Fold arbitrary Miller indices into the stored half grid.

    An index whose third component falls outside 0..Nyquist is not stored; its value
    is the conjugate of the stored value at the negated index. The caller must apply
    that conjugation, which is what ``sym_conj`` records for the symmetry arrays.

    Args:
        j0, j1, j2: Miller indices, any sign, any magnitude.
        shape: Real-space grid shape.

    Returns:
        The three half-grid indices and a boolean array that is True where the
        fetched value must be conjugated.
    """
    n0, n1, n2 = shape
    j2m = np.mod(j2, n2)
    need = j2m > n2 // 2
    i0 = np.where(need, np.mod(-j0, n0), np.mod(j0, n0))
    i1 = np.where(need, np.mod(-j1, n1), np.mod(j1, n1))
    i2 = np.where(need, n2 - j2m, j2m)
    return i0, i1, i2, need


def flat_index(i0, i1, i2, shape):
    """Turn half-grid indices into offsets into the raveled array.

    ``core.project_sym`` gathers from a flattened spectrum, so the symmetry arrays hold
    flat offsets rather than triples.

    Args:
        i0, i1, i2: Half-grid indices, already folded.
        shape: Real-space grid shape.

    Returns:
        The flat offsets as int32.
    """
    H = h_shape(shape)
    return ((i0 * H[1] + i1) * H[2] + i2).astype(np.int32)


def index_matrix(op):
    """Give the matrix by which an operator permutes Miller indices.

    Recovered by applying the operator to the three unit indices. Used to expand the
    measured reflections over their symmetry orbit; the symmetry *arrays* need
    ``miller_matrix`` instead, which is its inverse transpose.

    Args:
        op: A ``gemmi.Op``.

    Returns:
        The integer 3x3 matrix ``M`` with ``h' = M h``.
    """
    cols = [op.apply_to_hkl([1, 0, 0]), op.apply_to_hkl([0, 1, 0]), op.apply_to_hkl([0, 0, 1])]
    return np.array(cols, dtype=np.int64).T


def miller_matrix(op):
    """Give the matrix by which a symmetry operator acts on Miller indices.

    For a real-space operator ``x -> R x + t``, a symmetric map's spectrum satisfies
    ``F(h) = exp(2i pi (M h).t) F(M h)`` with ``M = R^-T``. This returns that ``M``.

    Args:
        op: A ``gemmi.Op``.

    Returns:
        The integer 3x3 matrix ``R^-T``.

    Raises:
        PrepareError: If ``R^-T`` is not integral, which would mean the operator does
            not map the reciprocal lattice to itself.
    """
    rot = np.array(op.rot, dtype=np.float64) / gemmi.Op.DEN
    inverse_transpose = np.linalg.inv(rot).T
    matrix = np.rint(inverse_transpose).astype(np.int64)
    if not np.allclose(matrix, inverse_transpose, atol=1e-9):
        raise PrepareError(f"operator {op.triplet()} has a non-integral R^-T")
    return matrix


def assert_closed_under_inverse(ops):
    """Check the operator set contains the inverse of every member.

    Under the multiplicity-weighted inner product the adjoint of one operator is the
    operator of the inverse group element, so ``project_sym`` averaged over an
    inverse-closed set is self-adjoint, and over any other set is not.

    Args:
        ops: The space group operators, centring translations included.

    Returns:
        The number of distinct operators.

    Raises:
        PrepareError: If any inverse is missing.
    """
    triplets = {op.wrap().triplet() for op in ops}
    missing = [op.triplet() for op in ops if op.inverse().wrap().triplet() not in triplets]
    if missing:
        raise PrepareError(
            f"operator set is not closed under inverse; missing inverses of {missing}. "
            "project_sym is self-adjoint only for an inverse-closed set"
        )
    return len(triplets)


def symmetry_arrays(spacegroup, shape):
    """Build the gather, phase and conjugation arrays ``core.project_sym`` needs.

    For each operator and each half-grid point, evaluates ``M h``, folds it into the
    stored half grid, and records the flat source index, the phase
    ``exp(2i pi (M h).t)``, and whether the fold requires a conjugate. The set is
    checked for closure under inverse first.

    Args:
        spacegroup: The ``gemmi.SpaceGroup``.
        shape: Real-space grid shape.

    Returns:
        ``(sym_idx, sym_phase, sym_conj)``, each of shape ``(K,) + h_shape(shape)``.

    Raises:
        PrepareError: If the operator set is not closed under inverse, or an operator
            does not act integrally on Miller indices.
    """
    ops = list(spacegroup.operations())
    assert_closed_under_inverse(ops)
    h0, h1, h2 = signed_miller(shape)
    idx, phase, conj = [], [], []
    for op in ops:
        matrix = miller_matrix(op)
        tran = np.array(op.tran, dtype=np.float64) / gemmi.Op.DEN
        j0 = matrix[0, 0] * h0 + matrix[0, 1] * h1 + matrix[0, 2] * h2
        j1 = matrix[1, 0] * h0 + matrix[1, 1] * h1 + matrix[1, 2] * h2
        j2 = matrix[2, 0] * h0 + matrix[2, 1] * h1 + matrix[2, 2] * h2
        i0, i1, i2, need = fold_indices(j0, j1, j2, shape)
        idx.append(flat_index(i0, i1, i2, shape))
        phase.append(np.exp(2j * np.pi * (j0 * tran[0] + j1 * tran[1] + j2 * tran[2])))
        conj.append(need)
    return (
        jnp.asarray(np.stack(idx).astype(np.int32)),
        jnp.asarray(np.stack(phase).astype(np.complex128)),
        jnp.asarray(np.stack(conj).astype(bool)),
    )


def multiplicity(shape):
    """Build the Friedel multiplicity of the half grid.

    A coefficient off the self-conjugate planes stands for two physical reflections,
    since its Friedel mate is not stored; one on the ``l = 0`` or Nyquist plane
    stands for one, because both members of the pair are stored. Every
    reciprocal-space sum in ``core`` carries this factor.

    Args:
        shape: Real-space grid shape.

    Returns:
        An int32 array of 1s and 2s with shape ``h_shape(shape)``.
    """
    H = h_shape(shape)
    mult = np.full(H, 2, dtype=np.int32)
    mult[:, :, 0] = 1
    if shape[2] % 2 == 0:
        mult[:, :, shape[2] // 2] = 1
    return mult


def projector_residuals(sym_idx, sym_phase, sym_conj, shape, mult, seed=0):
    """Measure how far the symmetry projector is from being a projector.

    Args:
        sym_idx, sym_phase, sym_conj: The symmetry arrays.
        shape: Real-space grid shape.
        mult: Friedel multiplicity, which weights the inner product.
        seed: PRNG key for the random spectra used as probes.

    Returns:
        Tuple of the relative idempotence residual and the relative
        self-adjointness residual, both measured under the multiplicity-weighted
        inner product.
    """
    rng = np.random.default_rng(seed)
    F = core.r2c(jnp.asarray(rng.standard_normal(shape)))
    G = core.r2c(jnp.asarray(rng.standard_normal(shape)))
    P = core.project_sym(F, sym_idx, sym_phase, sym_conj)
    PP = core.project_sym(P, sym_idx, sym_phase, sym_conj)
    PG = core.project_sym(G, sym_idx, sym_phase, sym_conj)
    scale = float(jnp.max(jnp.abs(P)))
    idempotence = float(jnp.max(jnp.abs(PP - P))) / scale if scale else 0.0
    weights = jnp.asarray(mult)
    lhs = float(jnp.sum(weights * jnp.real(P * jnp.conj(G))))
    rhs = float(jnp.sum(weights * jnp.real(F * jnp.conj(PG))))
    denominator = max(abs(lhs), abs(rhs))
    self_adjointness = abs(lhs - rhs) / denominator if denominator else 0.0
    return idempotence, self_adjointness


def assert_projector(sym_idx, sym_phase, sym_conj, shape, mult, tol=PROJECTOR_TOL):
    """Require the symmetry projector to be idempotent and self-adjoint.

    Run on every build **and on every read**, so a hand-edited or stale metadata file
    fails at load rather than silently refining against the wrong symmetry. There is
    no allowlist of supported space groups: a group either passes this or the run
    stops.

    Args:
        sym_idx, sym_phase, sym_conj: The symmetry arrays.
        shape: Real-space grid shape.
        mult: Friedel multiplicity.
        tol: Bound on both residuals.

    Returns:
        The two measured residuals, which are recorded in the metadata.

    Raises:
        PrepareError: If either residual exceeds the tolerance. Fix the arrays, never
            the tolerance.
    """
    idempotence, self_adjointness = projector_residuals(
        sym_idx, sym_phase, sym_conj, shape, mult
    )
    if idempotence > tol or self_adjointness > tol:
        raise PrepareError(
            f"project_sym for this space group is not a projector: "
            f"idempotence {idempotence:.3e}, self-adjointness {self_adjointness:.3e}, "
            f"tolerance {tol:.0e}"
        )
    return idempotence, self_adjointness


def read_model(cif_path):
    """Read a coordinate file and set up its entities.

    Args:
        cif_path: Path to the coordinate mmCIF.

    Returns:
        The ``gemmi.Structure``. Not called at all in zero-field mode, where no model
        is read.
    """
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    return st


def density_map(st, d_min, shape, sample_rate=SAMPLE_RATE, cutoff=DENSITY_CUTOFF):
    """Compute the model's electron density on the pipeline grid.

    Uses gemmi's REFMAC-compatible blur, which smears the atoms so a coarse grid can
    carry them; the smearing is undone in reciprocal space by ``exp(blur s^2 / 4)``.
    At sample_rate 3 the blur is worth two orders of magnitude of accuracy.

    Args:
        st: The model structure.
        d_min: Resolution limit.
        shape: The pipeline grid shape.
        sample_rate: Grid points per d_min; halved to reach gemmi's rate convention.
        cutoff: Atom density truncation; 1e-7 rather than gemmi's 1e-5 default.

    Returns:
        Tuple of the density as **float64** -- gemmi grids are float32, and this is
        the one place the cast happens -- and the blur that must be undone.
    """
    dc = gemmi.DensityCalculatorX()
    dc.d_min = d_min
    dc.rate = sample_rate / 2.0
    dc.cutoff = cutoff
    dc.set_refmac_compatible_blur(st[0])
    dc.set_grid_cell_and_spacegroup(st)
    dc.grid.set_size(*shape)
    dc.add_model_density_to_grid(st[0])
    dc.grid.symmetrize_sum()
    # gemmi grids are float32; cast once, here
    return np.asarray(dc.grid.array, dtype=np.float64), float(dc.blur)


def solvent_mask_map(st, shape):
    """Compute the bulk-solvent mask on the pipeline grid.

    Args:
        st: The model structure.
        shape: The pipeline grid shape.

    Returns:
        The mask as **float64** -- gemmi's grid is float32, and this is one of the two
        places the cast happens. Its structure factors are scaled against the work set
        alongside the model's.
    """
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)
    grid = gemmi.FloatGrid(*shape)
    grid.set_unit_cell(st.cell)
    grid.spacegroup = st.find_spacegroup()
    masker.put_mask_on_float_grid(grid, st[0])
    return np.asarray(grid.array, dtype=np.float64)


def to_structure_factors(rho, cell):
    """Convert a real map to crystallographic structure factors.

    ``core.r2c`` is an orthonormal rfft in numpy's ``exp(-2i pi h.r)`` convention,
    while crystallography uses ``exp(+2i pi h.r)``, so
    ``F_xtal = (V / sqrt(N)) conj(r2c(rho))``. Getting either half of that wrong is
    what pipeline test 1a exists to catch.

    Args:
        rho: Real-space map.
        cell: Unit cell, for the volume.

    Returns:
        The half-spectrum structure factors as complex128.
    """
    rho = np.asarray(rho, dtype=np.float64)
    scale = cell.volume / np.sqrt(rho.size)
    return scale * np.conj(np.asarray(core.r2c(jnp.asarray(rho)), dtype=np.complex128))


def from_structure_factors(F, cell, shape):
    """Invert :func:`to_structure_factors`.

    Args:
        F: Crystallographic structure factors on the half grid.
        cell: Unit cell, for the volume.
        shape: Real-space grid shape.

    Returns:
        The real map as float64.
    """
    scale = np.sqrt(int(np.prod(shape))) / cell.volume
    spectrum = jnp.asarray(np.conj(np.asarray(F, dtype=np.complex128)) * scale)
    return np.asarray(core.c2r(spectrum, tuple(shape)), dtype=np.float64)


def inverse_d2(cell, shape):
    """Give 1/d^2 for every half-grid point.

    Args:
        cell: The unit cell.
        shape: Real-space grid shape.

    Returns:
        A float64 array of shape ``h_shape(shape)``. Drives the resolution-dependent
        terms: the solvent fall-off, the B factors, and the resolution shells.
    """
    h0, h1, h2 = signed_miller(shape)
    hkl = np.stack([h0.ravel(), h1.ravel(), h2.ravel()], axis=1).astype(np.int32)
    return cell.calculate_1_d2_array(hkl).reshape(h0.shape).astype(np.float64)


def free_flag_convention(counts):
    """Detect which flag value marks the test set, rather than assuming one.

    ``free == 1`` is not safe to assume. A binary column is resolved by taking the
    **rarer** class as the test set, whichever integer it carries; a contiguous
    0..N column with at least five bins is the CNS/CCP4 convention where bin 0 is
    the test set.

    Args:
        counts: Mapping of flag value to reflection count.

    Returns:
        Tuple of the free value, a sentence describing how it was resolved, and the
        per-class fractions. All three are recorded in the metadata.

    Raises:
        PrepareError: If the column is single-valued, if it matches neither shape, or
            if the resolved free set falls outside 2-15% of reflections -- too small
            to measure with, or too large to be a test set.
    """
    counts = {int(value): int(count) for value, count in counts.items()}
    total = sum(counts.values())
    if total == 0:
        raise PrepareError("the free-flag column is empty")
    fractions = {value: counts[value] / total for value in counts}
    values = sorted(counts)
    if len(values) < 2:
        raise PrepareError(
            f"free-flag column holds the single value {values[0]}; no test set was deposited"
        )
    if len(values) == 2:
        free_value = min(counts, key=lambda value: counts[value])
        convention = (
            f"binary column {values}, free = {free_value}, detected as the rarer class "
            f"({fractions[free_value]:.2%} of reflections)"
        )
    elif values == list(range(len(values))) and len(values) >= 5:
        free_value = 0
        convention = (
            f"CNS/CCP4 {len(values)}-bin column 0..{values[-1]}, free = 0 by that convention "
            f"({fractions[0]:.2%} of reflections)"
        )
    else:
        raise PrepareError(
            f"cannot determine the free-flag convention from distinct values {values} "
            f"with fractions { {v: round(f, 4) for v, f in fractions.items()} }; "
            "neither a binary column nor a contiguous 0..N bin column"
        )
    fraction = fractions[free_value]
    low, high = FREE_FRACTION_RANGE
    if not low <= fraction <= high:
        raise PrepareError(
            f"free set is {fraction:.2%} of reflections, outside the sane range "
            f"{low:.0%}-{high:.0%}; resolved convention was {convention!r}. "
            "A test set this size makes delta_R_free either noisy or leaky"
        )
    return free_value, convention, fractions


def reciprocal_arrays(mtz, shape, labels, free_value, spacegroup):
    """Place the measured reflections onto the half grid.

    Each unique reflection is expanded over its full symmetry orbit and over Friedel
    mates, which is what makes ``mask_obs`` invariant under every operator -- without
    that, ``project_band`` and ``project_sym`` would not commute and ``pi_real``
    would not be a projector.

    Args:
        mtz: Reflection file.
        shape: Real-space grid shape.
        labels: Dict with the ``f``, ``sig`` and ``free`` column labels.
        free_value: The flag value that marks the test set.
        spacegroup: The space group whose orbits are used for the expansion.

    Returns:
        Dict with ``amp``, ``wgt`` (1/sigma^2), the three masks, and the counts
        ``n_work``, ``n_free``, ``n_dropped`` and ``n_unique``.

    Raises:
        PrepareError: If the grid is too coarse for the measured resolution, if two
            unique reflections land on one grid point with different amplitudes, or
            if the filled points disagree with ``mask_obs``.
    """
    H = h_shape(shape)
    amp = np.zeros(H, dtype=np.float64)
    wgt = np.zeros(H, dtype=np.float64)
    work = np.zeros(H, dtype=bool)
    free = np.zeros(H, dtype=bool)
    filled = np.zeros(H, dtype=bool)

    hkl = mtz.make_miller_array().astype(np.int64)
    f_obs = np.asarray(mtz.column_with_label(labels["f"]).array, dtype=np.float64)
    sigma = np.asarray(mtz.column_with_label(labels["sig"]).array, dtype=np.float64)
    flag = np.asarray(mtz.column_with_label(labels["free"]).array, dtype=np.float64)

    usable = np.isfinite(f_obs) & np.isfinite(sigma) & (sigma > 0) & (f_obs > 0)
    dropped = int((~usable).sum())
    hkl, f_obs, sigma, flag = hkl[usable], f_obs[usable], sigma[usable], flag[usable]
    is_free = np.rint(flag).astype(np.int64) == free_value

    limit = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2], dtype=np.int64)
    ops = list(spacegroup.operations())
    for op in ops:
        matrix = index_matrix(op)
        for sign in (1, -1):
            equivalent = sign * (hkl @ matrix.T)
            if np.any(np.abs(equivalent) >= limit):
                raise PrepareError(
                    "grid is too coarse for the measured resolution: "
                    f"|h| up to {np.abs(equivalent).max(axis=0)} against limit {limit}"
                )
            i0, i1, i2, _ = fold_indices(
                equivalent[:, 0], equivalent[:, 1], equivalent[:, 2], shape
            )
            clash = filled[i0, i1, i2] & (np.abs(amp[i0, i1, i2] - f_obs) > 1e-6 * f_obs)
            if np.any(clash):
                raise PrepareError(
                    f"{int(clash.sum())} grid points receive two different amplitudes; "
                    "the reflection list is inconsistent with the space group"
                )
            amp[i0, i1, i2] = f_obs
            wgt[i0, i1, i2] = 1.0 / sigma ** 2
            filled[i0, i1, i2] = True
            work[i0, i1, i2] = ~is_free
            free[i0, i1, i2] = is_free

    obs = work | free
    if not np.array_equal(obs, filled):
        raise PrepareError("mask_obs disagrees with the set of filled grid points")
    return {
        "amp": amp, "wgt": wgt, "mask_work": work, "mask_free": free, "mask_obs": obs,
        "n_work": int(work.sum()), "n_free": int(free.sum()), "n_dropped": dropped,
        "n_unique": int(hkl.shape[0]),
    }


def scale_design(h0, h1, h2):
    """Build the design matrix of the log-scale fit.

    Columns are ``[1, -h^2, -k^2, -l^2, -2hk, -2hl, -2kl]``, so a least-squares fit in
    log space gives an overall scale plus the six components of an anisotropic B.

    Args:
        h0, h1, h2: Miller indices, of any common shape.

    Returns:
        The design matrix, with the seven columns on the trailing axis.
    """
    ones = np.ones(h0.shape, dtype=np.float64)
    return np.stack(
        [ones, -(h0 * h0), -(h1 * h1), -(h2 * h2),
         -2.0 * h0 * h1, -2.0 * h0 * h2, -2.0 * h1 * h2],
        axis=-1,
    ).astype(np.float64)


def fit_scales(f_calc, f_mask, amp, wgt, mask_work, s2, shape,
               k_grid=SOLVENT_K, b_grid=SOLVENT_B):
    """Fit the overall scale, anisotropic B and bulk solvent **on work only**.

    Fits ``log(F_obs / |F_model|)`` against a quadratic in the Miller indices by
    weighted least squares, scanning a small grid of solvent parameters and keeping
    the lowest weighted residual.

    Free reflections enter none of it. A scale fitted across all measured reflections
    couples work to free and produces a small non-zero delta_R_free that looks like a
    result -- the single most likely cause of a failing null gate.

    Args:
        f_calc: Model structure factors, unblurred.
        f_mask: Solvent mask structure factors.
        amp: Observed amplitudes.
        wgt: Weights, 1/sigma^2.
        mask_work: The work set. Nothing outside it is read.
        s2: 1/d^2 per grid point.
        shape: Real-space grid shape.
        k_grid, b_grid: Solvent scale and B values to scan.

    Returns:
        Dict with the log-scale ``coefficients``, ``k_sol``, ``b_sol`` and the
        weighted ``residual``.

    Raises:
        PrepareError: If no work reflections are available, or no solvent parameters
            give a usable model.
    """
    h0, h1, h2 = signed_miller(shape)
    selected = mask_work & (amp > 0)
    if not selected.any():
        raise PrepareError("no work reflections available for the scale fit")
    design = scale_design(h0[selected], h1[selected], h2[selected])
    amp_w, wgt_w, s2_w = amp[selected], wgt[selected], s2[selected]
    best = None
    for k_sol in k_grid:
        for b_sol in b_grid:
            total = f_calc[selected] + k_sol * np.exp(-b_sol * s2_w / 4.0) * f_mask[selected]
            magnitude = np.abs(total)
            good = magnitude > 0
            if not good.any():
                continue
            weight = wgt_w[good] * amp_w[good] ** 2
            root = np.sqrt(weight)
            target = np.log(amp_w[good] / magnitude[good])
            coefficients, *_ = np.linalg.lstsq(
                design[good] * root[:, None], target * root, rcond=None
            )
            model = np.exp(design[good] @ coefficients) * magnitude[good]
            residual = float(np.sum(wgt_w[good] * (amp_w[good] - model) ** 2))
            if best is None or residual < best["residual"]:
                best = {
                    "residual": residual, "coefficients": coefficients,
                    "k_sol": float(k_sol), "b_sol": float(b_sol),
                }
    if best is None:
        raise PrepareError("the scale fit found no usable solvent parameters")
    return best


def scale_field(coefficients, shape):
    """Evaluate the fitted scale over the whole half grid.

    The coefficients come from a fit on the work set, but the scale they define is
    applied everywhere -- parameters from work only, applied to all reflections, is
    exactly what the invariant asks for.

    Args:
        coefficients: The seven log-scale coefficients.
        shape: Real-space grid shape.

    Returns:
        The multiplicative scale per grid point, as float64.
    """
    h0, h1, h2 = signed_miller(shape)
    return np.exp(scale_design(h0, h1, h2) @ coefficients).astype(np.float64)


def assert_float64(ctx):
    """Require every context array to be double precision or an exact type.

    gemmi's grids are float32; the cast to float64 happens once, at the adapter
    boundary, and this asserts nothing single-precision survived into the context.

    Args:
        ctx: The assembled ``core.Ctx``.

    Returns:
        True.

    Raises:
        PrepareError: Naming the first field with a disallowed dtype.
    """
    for name, value in zip(type(ctx)._fields, ctx):
        dtype = np.asarray(value).dtype
        if dtype.type not in FLOAT64_DTYPES:
            raise PrepareError(f"Ctx.{name} has dtype {dtype}; the pipeline is float64 only")
    return True


def ctx_paths(pdb_id, root="data", zero_field=False):
    """Give the context and metadata paths for one entry and mode.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        zero_field: Select the zero-field pair.

    Returns:
        Dict with the ``ctx`` and ``meta`` paths. Model and zero-field runs never share
        a file, so one can never silently be read in place of the other.
    """
    paths = fetch.cache_paths(pdb_id, root)
    suffix = "_zero" if zero_field else ""
    return {
        "ctx": paths["dir"] / f"ctx{suffix}.npz",
        "meta": paths["dir"] / f"meta{suffix}.json",
    }


def write_ctx(ctx, path):
    """Write a context, omitting the arrays the loader can rebuild.

    ``sym_idx``, ``sym_phase`` and ``sym_conj`` are derived from the space group and
    the grid, and for a multi-operator group outweigh everything else in the context,
    so they are left out and rebuilt on read.

    Args:
        ctx: The ``core.Ctx``.
        path: Destination ``.npz``.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray(value)
        for name, value in zip(type(ctx)._fields, ctx)
        if name not in DERIVED_FIELDS
    }
    np.savez_compressed(path, **arrays)
    return path


def read_ctx(path, meta):
    """Load a context, rebuilding the symmetry arrays it deliberately omits.

    ``sym_idx``, ``sym_phase`` and ``sym_conj`` are derived data and for a
    multi-operator group are larger than the rest of the context put together, so
    they are not stored. They are rebuilt here from the space group and grid in the
    metadata, and the projector checks run again on every read.

    Args:
        path: Path to the ``.npz``.
        meta: The matching metadata, for ``spacegroup`` and ``grid``.

    Returns:
        The ``core.Ctx``.

    Raises:
        PrepareError: If fields are missing, the space group is unknown, the
            projector checks fail, or any array is single precision.
    """
    with np.load(path) as data:
        fields = {name: jnp.asarray(data[name]) for name in data.files}
    missing = set(core.Ctx._fields) - set(fields) - set(DERIVED_FIELDS)
    if missing:
        raise PrepareError(f"{path} is missing Ctx fields {sorted(missing)}")
    shape = tuple(meta["grid"])
    spacegroup = gemmi.find_spacegroup_by_name(meta["spacegroup"])
    if spacegroup is None:
        spacegroup = gemmi.find_spacegroup_by_number(meta["spacegroup_number"])
    if spacegroup is None:
        raise PrepareError(f"metadata names an unknown space group {meta['spacegroup']!r}")
    sym_idx, sym_phase, sym_conj = symmetry_arrays(spacegroup, shape)
    # rebuilt, never stored; checked on every read
    assert_projector(sym_idx, sym_phase, sym_conj, shape, np.asarray(fields["mult"]))
    fields.update(sym_idx=sym_idx, sym_phase=sym_phase, sym_conj=sym_conj)
    ctx = core.Ctx(**fields)
    assert_float64(ctx)
    return ctx


def write_metadata(meta, path):
    """Write the run metadata as JSON.

    The metadata is not a log: ``read_ctx`` rebuilds the symmetry arrays from its
    ``spacegroup`` and ``grid``, so editing it by hand changes what is refined.

    Args:
        meta: The metadata dict.
        path: Destination path.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def read_metadata(path):
    """Read run metadata written by :func:`write_metadata`.

    Args:
        path: Path to the JSON file.

    Returns:
        The metadata dict.
    """
    return json.loads(Path(path).read_text())


def build_ctx(info, sample_rate=SAMPLE_RATE, cutoff=DENSITY_CUTOFF, sqrt_s="flat",
              s_min=S_MIN, rho_floor=RHO_FLOOR, mu=0.0, zero_field=False):
    """Build a context from a fetched entry, in model or zero-field mode.

    The order matters: grid, then density and solvent, then the work-only scale fit,
    then the reciprocal arrays, then symmetry, then the assertions. In zero-field
    mode no model is read at all -- rho0 is exactly zero, there is no solvent mask
    and no scale fit, and cell and symmetry come from the reflections.

    Args:
        info: A ``fetch.fetch`` result.
        sample_rate: Grid points per d_min.
        cutoff: Atom density truncation.
        sqrt_s: Stationary spectrum for z; only ``"flat"`` is implemented.
        s_min: Floor of the sigma field.
        rho_floor: Density below which the positivity term acts.
        mu: Weight of the positivity term.
        zero_field: Refine from nothing rather than from the deposited model.

    Returns:
        Tuple of the ``core.Ctx`` and its metadata.

    Raises:
        PrepareError: From any of the stage's checks -- space group disagreement,
            free-flag convention, orbit expansion, projector, mask invariance,
            dtype, or a non-finite rho0.
    """
    mtz = fetch.read_mtz(info["mtz"])
    if zero_field:
        # no model is read at all: cell and symmetry come from the reflections
        spacegroup = mtz.spacegroup
        cell = mtz.cell
    else:
        st = read_model(info["cif"])
        spacegroup = st.find_spacegroup()
        if spacegroup is None:
            raise PrepareError(f"{info['id']}: the model carries no space group")
        if spacegroup.xhm() != mtz.spacegroup.xhm():
            raise PrepareError(
                f"model and MTZ disagree on the space group: "
                f"{spacegroup.xhm()!r} against {mtz.spacegroup.xhm()!r}"
            )
        cell = st.cell
    if spacegroup is None:
        raise PrepareError(f"{info['id']}: the MTZ carries no space group")
    shape = grid_shape(mtz, sample_rate)
    d_min = mtz.resolution_high()
    s2 = inverse_d2(cell, shape)

    free_value, convention, fractions = free_flag_convention(info["free_flag_counts"])
    labels = {"f": info["f_label"], "sig": info["sig_label"], "free": info["free_label"]}
    data = reciprocal_arrays(mtz, shape, labels, free_value, spacegroup)

    if zero_field:
        blur = 0.0
        fit = {"coefficients": np.zeros(7), "k_sol": 0.0, "b_sol": 0.0, "residual": float("nan")}
        rho0 = np.zeros(shape, dtype=np.float64)
        rho0_source = "zeros: no model, no bulk solvent, no scale fit"
    else:
        rho_atoms, blur = density_map(st, d_min, shape, sample_rate, cutoff)
        mask_map = solvent_mask_map(st, shape)
        f_calc = to_structure_factors(rho_atoms, cell) * np.exp(blur * s2 / 4.0)
        f_mask = to_structure_factors(mask_map, cell)
        fit = fit_scales(f_calc, f_mask, data["amp"], data["wgt"], data["mask_work"], s2, shape)
        solvent = fit["k_sol"] * np.exp(-fit["b_sol"] * s2 / 4.0)
        f_model = scale_field(fit["coefficients"], shape) * (f_calc + solvent * f_mask)
        rho0 = from_structure_factors(f_model, cell, shape)
        rho0_source = "gemmi DensityCalculatorX + SolventMasker, scaled on mask_work"

    sym_idx, sym_phase, sym_conj = symmetry_arrays(spacegroup, shape)
    mult = multiplicity(shape)
    idempotence, self_adjointness = assert_projector(
        sym_idx, sym_phase, sym_conj, shape, mult
    )
    obs = jnp.asarray(data["mask_obs"])
    for k in range(sym_idx.shape[0]):
        if not bool(jnp.all(obs.ravel()[sym_idx[k]] == obs)):
            raise PrepareError(
                f"mask_obs is not invariant under symmetry operator {k}; "
                "project_band and project_sym would not commute"
            )
    if sqrt_s != "flat":
        raise PrepareError(f"unknown sqrtS choice {sqrt_s!r}")
    sqrt_s_array = np.ones(h_shape(shape), dtype=np.float64)

    ctx = core.Ctx(
        rho0=jnp.asarray(rho0),
        F0=core.r2c(jnp.asarray(rho0)),
        amp=jnp.asarray(data["amp"]),
        wgt=jnp.asarray(data["wgt"]),
        mult=jnp.asarray(mult),
        mask_work=jnp.asarray(data["mask_work"]),
        mask_free=jnp.asarray(data["mask_free"]),
        mask_obs=obs,
        sqrtS=jnp.asarray(sqrt_s_array),
        sqrtS_sigma=jnp.zeros(h_shape(shape), dtype=jnp.float64),
        sym_idx=sym_idx, sym_phase=sym_phase, sym_conj=sym_conj,
        s_min=jnp.asarray(s_min, dtype=jnp.float64),
        rho_floor=jnp.asarray(rho_floor, dtype=jnp.float64),
        mu=jnp.asarray(mu, dtype=jnp.float64),
    )
    assert_float64(ctx)
    if not bool(jnp.all(jnp.isfinite(ctx.rho0))):
        raise PrepareError("rho0 is not finite")
    if not bool(jnp.all(ctx.mask_obs == (ctx.mask_work | ctx.mask_free))):
        raise PrepareError("mask_obs != mask_work | mask_free")

    meta = {
        "pdb_id": info["id"],
        "zero_field": bool(zero_field),
        "rho0_source": rho0_source,
        "spacegroup": spacegroup.xhm(),
        "spacegroup_number": spacegroup.number,
        "n_operators": int(sym_idx.shape[0]),
        "cell": [float(x) for x in cell.parameters],
        "cell_volume": float(cell.volume),
        "d_min": float(d_min),
        "d_max": float(mtz.resolution_low()),
        "grid": list(shape),
        "sample_rate": float(sample_rate),
        "density_cutoff": float(cutoff),
        "refmac_blur": float(blur),
        "free_label": info["free_label"],
        "free_value": int(free_value),
        "free_convention": convention,
        "free_fraction": float(fractions[free_value]),
        "free_flag_counts": {str(k): v for k, v in info["free_flag_counts"].items()},
        "free_flag_fractions": {str(k): round(float(v), 6) for k, v in fractions.items()},
        "n_work_grid": data["n_work"],
        "n_free_grid": data["n_free"],
        "n_unique_used": data["n_unique"],
        "n_reflections_dropped": data["n_dropped"],
        "scale_log_coefficients": [float(x) for x in fit["coefficients"]],
        "k_sol": fit["k_sol"],
        "b_sol": fit["b_sol"],
        "scale_fitted_on": "mask_work only",
        "sigma_a_estimated_on": "work reflections only",
        "sqrtS": sqrt_s,
        "s_min": float(s_min),
        "rho_floor": float(rho_floor),
        "projector_idempotence": idempotence,
        "projector_self_adjointness": self_adjointness,
    }
    return ctx, meta


def prepare(pdb_id, root="data", sample_rate=SAMPLE_RATE, force=False, zero_field=False,
            **kwargs):
    """Fetch, build and cache a context, or read back a matching cached one.

    The cache is keyed on the mode and the sampling rate, so changing either rebuilds
    rather than returning a stale context.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        sample_rate: Grid points per d_min.
        force: Rebuild even when a matching cache exists.
        zero_field: Build the zero-field context, cached separately.
        **kwargs: Passed through to :func:`build_ctx`.

    Returns:
        Tuple of the ``core.Ctx`` and its metadata.
    """
    info = fetch.fetch(pdb_id, root)
    paths = ctx_paths(pdb_id, root, zero_field)
    if not force and fetch.is_cached(paths["ctx"]) and fetch.is_cached(paths["meta"]):
        cached = read_metadata(paths["meta"])
        if (cached.get("sample_rate") == float(sample_rate)
                and cached.get("zero_field") == bool(zero_field)):
            return read_ctx(paths["ctx"], cached), cached
    ctx, meta = build_ctx(info, sample_rate=sample_rate, zero_field=zero_field, **kwargs)
    write_ctx(ctx, paths["ctx"])
    write_metadata(meta, paths["meta"])
    return ctx, meta
