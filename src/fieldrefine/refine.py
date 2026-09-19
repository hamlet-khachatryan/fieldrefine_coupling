"""Stage 4: the parameter grid. Exhaustive, no objective, one point in memory at a time.

Memory discipline, because the grid is where it matters:

* one grid point is live at any moment; its state is written and dropped before the next starts;
* ``core.run`` is a ``lax.scan`` over a donated ``step``, so the trajectory is never materialised;
* the scan runs in chunks so the trace stays bounded and the stopping rule can be checked;
* the trace is four scalars per step, not four grids.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import core

from . import fetch, nullgate, prepare

GRID_C = (0.0, 0.1, 0.2, 0.4, 0.8)
GRID_Q = (2.0, 4.0, 8.0, 16.0)
GRID_MU = (0.0, 1e-2, 1e-1, 1.0, 10.0)
MAX_STEPS = 2000
CHUNK = 100


class RefineError(RuntimeError):
    pass


def grid_spec():
    """Give the knob grid as the tuple ``core.knob_grid`` expects.

    Returns:
        ``(c values, q_sigma values, mu values)``.
    """
    return (GRID_C, GRID_Q, GRID_MU)


def n_points():
    """Count the grid points.

    Returns:
        The size of the Cartesian product of the three knob axes.
    """
    return len(GRID_C) * len(GRID_Q) * len(GRID_MU)


def knobs(index):
    """Map a flat index to one grid point.

    The mapping is C order -- c is the slowest axis, mu the fastest -- so a SLURM
    array index addresses a point directly.

    Args:
        index: Flat grid index.

    Returns:
        A ``core.Knobs`` with ``c``, ``q_sigma`` and ``mu``.
    """
    return core.knob_grid(index, grid_spec())


def null_index():
    """Locate the null point, c = 0 and mu = 0.

    Returns:
        Its flat index.

    Raises:
        RefineError: If the grid has no null point. It must always contain one.
    """
    for index in range(n_points()):
        point = knobs(index)
        if point.c == 0.0 and point.mu == 0.0:
            return index
    raise RefineError("the grid contains no null point; c = 0, mu = 0 must appear")


def epsilon_floor(ctx):
    """Estimate the noise floor of the data term from the deposited sigmas.

    Since ``wgt = 1/SIGF**2``, a fit agreeing with the data to within the reported
    sigmas has ``e_work = sum(mult * wgt * (k|F| - amp)**2)`` of about ``sum(mult)``.
    Reaching that means the residuals are the size the experiment says they should
    be; going below it is fitting noise. This is the **only** stopping rule --
    R_free is never consulted.

    Args:
        ctx: The context, whose ``mask_work`` and ``mult`` set the floor.

    Returns:
        Tuple of the floor value and the name of the estimator, both recorded per
        grid point.
    """
    # at the noise floor each work term contributes mult * wgt * sigma^2 = mult
    value = float(jnp.sum(jnp.where(ctx.mask_work, ctx.mult, 0)))
    return value, "chi2_unity: sum(mult) over mask_work, from the deposited SIGF"


def grid_dir(pdb_id, root="data", zero_field=False):
    """Give the directory holding one run's grid points.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        zero_field: Select the zero-field tree.

    Returns:
        The directory path; model and zero-field grids are kept apart.
    """
    directory = fetch.cache_paths(pdb_id, root)["dir"]
    return directory / ("grid_zero" if zero_field else "grid")


def point_paths(pdb_id, index, root="data", zero_field=False):
    """Give the output paths for one grid point.

    Each point writes its own files, so a job array has nothing to contend over.

    Args:
        pdb_id: PDB entry code.
        index: Flat grid index.
        root: Cache root directory.
        zero_field: Select the zero-field tree.

    Returns:
        Dict with the ``dir``, the ``state`` ``.npz`` and the ``result`` JSON.
    """
    directory = grid_dir(pdb_id, root, zero_field)
    return {
        "dir": directory,
        "state": directory / f"point_{index:04d}.npz",
        "result": directory / f"point_{index:04d}.json",
    }


def write_point(state, trace, result, paths):
    """Write one grid point's state, trace and result.

    Called before the next point starts, which is what keeps peak memory independent
    of the number of points.

    Args:
        state: The refined state.
        trace: Its trace; four scalars per step, not four grids.
        result: The result dict, written as JSON beside the arrays.
        paths: A :func:`point_paths` mapping.

    Returns:
        The paths written.
    """
    paths["dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["state"],
        z=np.asarray(state.z), u=np.asarray(state.u),
        j=np.asarray(trace.j), e_work=np.asarray(trace.e_work),
        r_work=np.asarray(trace.r_work), r_free=np.asarray(trace.r_free),
    )
    Path(paths["result"]).write_text(json.dumps(result, indent=2, sort_keys=True))
    return paths


def read_point(paths):
    """Read back one grid point.

    Args:
        paths: A :func:`point_paths` mapping.

    Returns:
        Tuple of the state, the trace and the result dict. A finished point is re-read
        rather than recomputed, so an interrupted grid resumes.
    """
    with np.load(paths["state"]) as data:
        state = core.State(jnp.asarray(data["z"]), jnp.asarray(data["u"]))
        trace = core.Trace(
            jnp.asarray(data["j"]), jnp.asarray(data["e_work"]),
            jnp.asarray(data["r_work"]), jnp.asarray(data["r_free"]),
        )
    return state, trace, json.loads(Path(paths["result"]).read_text())


def point_ctx(ctx, point):
    """Specialise a context to one grid point's knobs.

    Sets the sigma spectrum from ``c`` and ``q_sigma`` and the positivity weight from
    ``mu``; everything else, the data and the symmetry included, is shared across the
    grid.

    Args:
        ctx: The prepared context.
        point: A ``core.Knobs``.

    Returns:
        The specialised context.
    """
    return ctx._replace(
        sqrtS_sigma=core.sigma_spectrum(point.c, point.q_sigma, ctx),
        mu=jnp.asarray(point.mu, dtype=jnp.float64),
    )


def require_gate(pdb_id, root="data", zero_field=False):
    """Refuse to run the grid unless the null gate passed for this mode.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        zero_field: Which mode's gate result to require.

    Returns:
        The gate result.

    Raises:
        RefineError: If no gate result exists, or if it did not pass. A grid run on
            a pipeline whose null does not hold produces numbers, not results.
    """
    path = nullgate.gate_paths(pdb_id, root, zero_field)["gate"]
    if not fetch.is_cached(path):
        raise RefineError(
            f"no null gate result at {path}; run the null gate before the grid"
        )
    result = nullgate.read_result(path)
    if not result.get("passed"):
        measured = {
            name: result.get(name) for name in
            ("delta_r_free", "nullspace_energy_fraction", "symmetry_residual")
        }
        raise RefineError(
            f"the null gate did not pass for {pdb_id}: {measured}. "
            "Do not run the grid until it does"
        )
    return result


def refine_point(ctx, index, lr=None, max_steps=MAX_STEPS, chunk=CHUNK, seed=0):
    """Refine one grid point to the noise floor or to the step cap.

    Runs ``core.run`` in chunks so the stopping rule can be tested between scans
    without breaking into the compiled loop, and so the trace held on the device
    stays bounded.

    Args:
        ctx: The prepared context.
        index: Flat grid index, mapped through :func:`knobs`.
        lr: Step size; estimated from J when None.
        max_steps: Cap on steps. A cap, not a criterion.
        chunk: Steps per scan between stopping-rule checks.
        seed: PRNG key for the zero-field start state.

    Returns:
        Tuple of the final state, the concatenated trace, and a result dict holding
        the knobs, lr, steps, ``stopped_on``, the floor and its estimator, and the
        final J, e_work, R_work and R_free.
    """
    point = knobs(index)
    ctx_point = point_ctx(ctx, point)
    state = nullgate.start_state(ctx_point, seed=seed)
    if lr is None:
        lr = nullgate.learning_rate(state, ctx_point)
    floor, estimator = epsilon_floor(ctx_point)
    columns = {"j": [], "e_work": [], "r_work": [], "r_free": []}
    steps, stopped = 0, "max_steps"
    while steps < max_steps:
        state, trace = core.run(state, ctx_point, lr, chunk)
        for name in columns:
            columns[name].append(np.asarray(getattr(trace, name)))
        steps += chunk
        if float(trace.e_work[-1]) <= floor:
            stopped = "epsilon_floor"
            break
    trace = core.Trace(**{
        name: jnp.asarray(np.concatenate(values)) for name, values in columns.items()
    })
    result = {
        "index": int(index),
        "c": float(point.c),
        "q_sigma": float(point.q_sigma),
        "mu": float(point.mu),
        "lr": float(lr),
        "steps": int(steps),
        "stopped_on": stopped,
        "epsilon_floor": floor,
        "epsilon_floor_estimator": estimator,
        "j_final": float(trace.j[-1]),
        "e_work_final": float(trace.e_work[-1]),
        "r_work_final": float(trace.r_work[-1]),
        "r_free_final": float(trace.r_free[-1]),
        "seed": int(seed),
    }
    return state, trace, result


def refine(pdb_id, indices=None, root="data", zero_field=False, lr=None, max_steps=MAX_STEPS,
           chunk=CHUNK, seed=0, force=False):
    """Refine grid points, one at a time, writing each before starting the next.

    Peak memory is therefore independent of how many points are run. Finished points
    are re-read rather than recomputed, so an interrupted grid resumes.

    Args:
        pdb_id: PDB entry code.
        indices: Points to run; None runs the whole grid.
        root: Cache root directory.
        zero_field: Refine from nothing rather than from the deposited model.
        lr: Step size; estimated per point from J when None.
        max_steps: Cap on steps per point.
        chunk: Steps per scan between stopping-rule checks.
        seed: PRNG key for the zero-field start state.
        force: Recompute points that are already on disk.

    Returns:
        One result dict per point, in the order requested.

    Raises:
        RefineError: If the null gate has not passed for this mode.
    """
    require_gate(pdb_id, root, zero_field)
    ctx, meta = prepare.prepare(pdb_id, root, zero_field=zero_field)
    if indices is None:
        indices = range(n_points())
    results = []
    for index in indices:
        paths = point_paths(pdb_id, index, root, zero_field)
        if not force and fetch.is_cached(paths["result"]) and fetch.is_cached(paths["state"]):
            results.append(json.loads(Path(paths["result"]).read_text()))
            continue
        state, trace, result = refine_point(
            ctx, index, lr=lr, max_steps=max_steps, chunk=chunk, seed=seed
        )
        result.update({"pdb_id": meta["pdb_id"], "zero_field": meta["zero_field"]})
        write_point(state, trace, result, paths)
        results.append(result)
        # drop the point before the next one starts
        del state, trace
    return results
