"""Stage 3: the null gate. Mandatory, and it runs before any science.

At ``c = 0`` and ``mu = 0`` the forward model cannot move a free reflection: sigma is constant, the
positivity term is off, and the data term is supported on ``mask_work``. Three numbers measure
whether that is true of the implementation as well as of the mathematics.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import core

from . import fetch, prepare

DELTA_R_FREE_TOL = 1e-8
NULLSPACE_TOL = 1e-12
SYMMETRY_TOL = 1e-12
N_STEPS = 100
Q_SIGMA = 2.0
LR_SAFETY = 0.5
POWER_ITERATIONS = 50

CAUSES = (
    "scale or bulk solvent fitted across mask_obs rather than mask_work",
    "mult missing or doubled",
    "a real-space operation applied outside delta_rho",
    "symmetry arrays wrong",
)


class NullGateError(RuntimeError):
    pass


def gate_paths(pdb_id, root="data", zero_field=False):
    """Give the location of the gate result for one entry and mode.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        zero_field: Select the zero-field result rather than the model-based one.

    Returns:
        Dict with the ``gate`` path; model and zero-field runs never share a file.
    """
    paths = fetch.cache_paths(pdb_id, root)
    suffix = "_zero" if zero_field else ""
    return {"gate": paths["dir"] / f"nullgate{suffix}.json"}


def null_ctx(ctx, q_sigma=Q_SIGMA):
    """Put a context into the null configuration.

    Sets ``c = 0``, which makes ``sqrtS_sigma`` exactly zero and sigma constant, and
    ``mu = 0``, which switches the positivity term off. With both, the data term is
    supported on ``mask_work`` and the model has no path to a free reflection.

    Args:
        ctx: The prepared context.
        q_sigma: Index radius passed to ``core.sigma_spectrum``; immaterial at c = 0.

    Returns:
        A copy of the context in the null configuration.
    """
    return ctx._replace(
        sqrtS_sigma=core.sigma_spectrum(0.0, q_sigma, ctx),
        mu=jnp.asarray(0.0, dtype=jnp.float64),
    )


def start_state(ctx, seed=0):
    """Build the starting state, seeding it only when there is no model.

    With a model, the start is ``z = u = 0``: the deposited model itself. With
    ``rho0`` identically zero every ``|F|`` would be zero and ``fit_scale`` would be
    0/0, so z is seeded from an explicit key at **unit scale** -- the prior on z is
    N(0, I) and ``fit_scale``'s k absorbs the amplitude scale, so scaling the seed to
    the data would make the prior swamp the data term and collapse z instead of
    fitting it. The seed's free components are projected out, so the starting model
    carries no test-set information.

    Args:
        ctx: The context, already in the null or a grid configuration.
        seed: PRNG key for the zero-field seed; recorded in the output.

    Returns:
        The starting ``core.State``.

    Raises:
        NullGateError: If the seeded start produces no amplitude on ``mask_work``.
    """
    shape = ctx.rho0.shape
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    if bool(jnp.any(ctx.rho0 != 0.0)):
        return core.State(zeros, zeros)
    # zero field: at z = 0 every |F| is 0 and fit_scale is 0/0, so seed z from an explicit key
    draw = jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float64)
    # unit scale: the prior on z is N(0, I) and fit_scale's k absorbs the amplitude scale
    seeded = core.c2r(core.r2c(draw) * ~ctx.mask_free, shape)
    state = core.State(seeded, zeros)
    power = float(jnp.sum(ctx.mask_work * jnp.abs(core.sf(core.rho_total(state, ctx), ctx)) ** 2))
    if not np.isfinite(power) or power <= 0.0:
        raise NullGateError("the seeded zero-field start produced no amplitude on mask_work")
    return state


def learning_rate(state, ctx, iterations=POWER_ITERATIONS, seed=0):
    """Estimate a stable step size from the curvature of J.

    Power-iterates the Hessian-vector product of the objective to approximate its
    largest eigenvalue, then returns ``0.5 / lambda_max``. It reads **J alone**:
    R_free never enters a hyperparameter, a stopping rule, or the objective.

    Args:
        state: The state at which curvature is measured.
        ctx: The context.
        iterations: Power iterations to run.
        seed: PRNG key for the starting vector.

    Returns:
        The step size as a float.

    Raises:
        NullGateError: If the estimate is not finite and positive.
    """
    def value(s):
        return core.objective(s, ctx)[0]

    hessian_product = jax.jit(lambda v: jax.jvp(jax.grad(value), (state,), (v,))[1])
    rng = np.random.default_rng(seed)
    vector = core.State(
        jnp.asarray(rng.standard_normal(state.z.shape)),
        jnp.asarray(rng.standard_normal(state.u.shape)),
    )
    largest = 0.0
    for _ in range(iterations):
        norm = jnp.sqrt(jnp.sum(vector.z ** 2) + jnp.sum(vector.u ** 2))
        vector = core.State(vector.z / norm, vector.u / norm)
        image = hessian_product(vector)
        largest = float(jnp.sqrt(jnp.sum(image.z ** 2) + jnp.sum(image.u ** 2)))
        vector = image
    if not np.isfinite(largest) or largest <= 0.0:
        raise NullGateError(f"Hessian norm estimate is {largest}; cannot choose a step size")
    # from J alone; R_free never enters a hyperparameter
    return LR_SAFETY / largest


def measure(state, ctx):
    """Measure the three null quantities at one state.

    Args:
        state: The state after the gate's run.
        ctx: The context used for that run.

    Returns:
        Dict with ``delta_r_free`` (R_free of the model minus R_free after the run,
        both at one work-fitted scale), ``nullspace_energy_fraction`` (the share of
        Delta-rho's power outside ``mask_obs``), ``symmetry_residual`` (the relative
        distance from Delta-rho to its symmetrised self, in the multiplicity-weighted
        norm, which equals the real-space L2 norm by Parseval), and ``delta_rho_norm``.
    """
    drho = core.delta_rho(state, ctx)
    spectrum = core.r2c(drho)
    power = ctx.mult * jnp.abs(spectrum) ** 2
    total = float(jnp.sum(power))
    outside = float(jnp.sum(jnp.where(ctx.mask_obs, 0.0, power)))
    projected = core.project_sym(spectrum, ctx.sym_idx, ctx.sym_phase, ctx.sym_conj)
    residual = float(jnp.sqrt(jnp.sum(ctx.mult * jnp.abs(projected - spectrum) ** 2)))
    norm = float(np.sqrt(total))
    return {
        "delta_r_free": float(core.delta_r_free(state, ctx)),
        "nullspace_energy_fraction": (outside / total) if total > 0.0 else 0.0,
        "symmetry_residual": (residual / norm) if norm > 0.0 else 0.0,
        "delta_rho_norm": norm,
    }


def run_gate(ctx, n_steps=N_STEPS, lr=None, q_sigma=Q_SIGMA, seed=0):
    """Run the refinement in the null configuration and measure the result.

    Args:
        ctx: The prepared context.
        n_steps: Steps to take before measuring.
        lr: Step size; estimated from J when None.
        q_sigma: Index radius for the sigma spectrum.
        seed: PRNG key for the zero-field start state.

    Returns:
        The three measured quantities, the per-check booleans and thresholds, the
        run parameters, J and R at the first and last trace rows, and ``passed``.
    """
    gated = null_ctx(ctx, q_sigma)
    state = start_state(gated, seed=seed)
    if lr is None:
        lr = learning_rate(state, gated)
    final, trace = core.run(state, gated, lr, n_steps)
    result = measure(final, gated)
    result.update({
        "lr": float(lr),
        "n_steps": int(n_steps),
        "q_sigma": float(q_sigma),
        "j_start": float(trace.j[0]),
        "j_end": float(trace.j[-1]),
        "r_work_start": float(trace.r_work[0]),
        "r_work_end": float(trace.r_work[-1]),
        "r_free_start": float(trace.r_free[0]),
        "r_free_end": float(trace.r_free[-1]),
    })
    result["checks"] = {
        "delta_r_free": abs(result["delta_r_free"]) < DELTA_R_FREE_TOL,
        "nullspace_energy_fraction": result["nullspace_energy_fraction"] < NULLSPACE_TOL,
        "symmetry_residual": result["symmetry_residual"] < SYMMETRY_TOL,
    }
    result["thresholds"] = {
        "delta_r_free": DELTA_R_FREE_TOL,
        "nullspace_energy_fraction": NULLSPACE_TOL,
        "symmetry_residual": SYMMETRY_TOL,
    }
    result["passed"] = all(result["checks"].values())
    return result


def assert_gate(result, pdb_id=None):
    """Raise unless every null check passed.

    Args:
        result: A ``run_gate`` result.
        pdb_id: Entry code, used in the message.

    Returns:
        The result unchanged, when it passed.

    Raises:
        NullGateError: Listing each measured value against its threshold and the four
            candidate causes, in the order worth checking. The thresholds are never
            to be loosened: a small non-zero value is a coupling to find.
    """
    if result["passed"]:
        return result
    subject = f" for {pdb_id}" if pdb_id else ""
    lines = [f"null gate FAILED{subject}:"]
    for name, ok in result["checks"].items():
        lines.append(
            f"  {name} = {result[name]:.6e}  threshold {result['thresholds'][name]:.0e}  "
            f"{'ok' if ok else 'FAIL'}"
        )
    lines.append("candidate causes, in the order worth checking:")
    lines.extend(f"  {n}. {cause}" for n, cause in enumerate(CAUSES, 1))
    lines.append("do not proceed to the grid; do not loosen the thresholds")
    raise NullGateError("\n".join(lines))


def write_result(result, path):
    """Write a gate result as JSON.

    Args:
        result: A ``run_gate`` result.
        path: Destination path.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return path


def read_result(path):
    """Read a gate result written by :func:`write_result`.

    Args:
        path: Path to the JSON file.

    Returns:
        The result dict.
    """
    return json.loads(Path(path).read_text())


def gate(pdb_id, root="data", n_steps=N_STEPS, lr=None, q_sigma=Q_SIGMA, raise_on_failure=True,
         zero_field=False, seed=0):
    """Prepare one entry, run the null gate, write the result.

    This is the mandatory checkpoint: nothing downstream is worth running until it
    passes, and ``refine`` refuses to start without it.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        n_steps: Steps to take before measuring.
        lr: Step size; estimated from J when None.
        q_sigma: Index radius for the sigma spectrum.
        raise_on_failure: Raise on failure rather than returning the numbers.
        zero_field: Gate the zero-field configuration instead of the model-based one.
        seed: PRNG key for the zero-field start state.

    Returns:
        The gate result, also written to ``data/<pdbid>/nullgate[_zero].json``.

    Raises:
        NullGateError: If any check failed and ``raise_on_failure`` is set.
    """
    ctx, meta = prepare.prepare(pdb_id, root, zero_field=zero_field)
    result = run_gate(ctx, n_steps=n_steps, lr=lr, q_sigma=q_sigma, seed=seed)
    result.update({
        "pdb_id": meta["pdb_id"],
        "zero_field": meta["zero_field"],
        "rho0_source": meta["rho0_source"],
        "spacegroup": meta["spacegroup"],
        "grid": meta["grid"],
        "n_work_grid": meta["n_work_grid"],
        "n_free_grid": meta["n_free_grid"],
    })
    write_result(result, gate_paths(pdb_id, root, zero_field)["gate"])
    if raise_on_failure:
        assert_gate(result, meta["pdb_id"])
    return result
