"""Stage 6: metrics. One tidy row per grid point, to CSV and to W&B.

``r_free`` is logged for the trace only. It is not an objective, and it must not be used to choose a
stopping point by eye either: that is selection by another route. The stopping rule is the noise
floor and nothing else.
"""

import json
from pathlib import Path

import gemmi
import jax.numpy as jnp
import numpy as np
import pandas as pd

import core

from . import fetch, maps, prepare, refine

WANDB_NOTE = (
    "r_free is logged per step for the trace only. It is not an objective and must not be used "
    "to choose a stopping point, by optimiser or by eye. The stopping rule is the noise floor."
)


class EvaluateError(RuntimeError):
    pass


def table_path(pdb_id, root="data", zero_field=False):
    """Give the CSV path for one entry and mode.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        zero_field: Select the zero-field table.

    Returns:
        The path; the per-shell JSON sits beside it with a ``.shells.json`` suffix.
    """
    directory = fetch.cache_paths(pdb_id, root)["dir"]
    return directory / ("evaluate_zero.csv" if zero_field else "evaluate.csv")


def shell_metrics(ctx, state, meta, n_shells=maps.N_SHELLS):
    """Compute R and correlation per resolution shell, for work and free.

    One scale, fitted on the work set, is applied to both -- the free shells are
    measured, never fitted.

    Args:
        ctx: The prepared context.
        state: The refined state.
        meta: Its metadata.
        n_shells: Requested shell count.

    Returns:
        A list of rows, each with ``set``, ``shell``, ``d_min``, ``n``, ``R``, ``CC``.
    """
    shape = tuple(meta["grid"])
    s2 = prepare.inverse_d2(gemmi.UnitCell(*meta["cell"]), shape)
    shells, n_shells = maps.shell_index(s2, ctx.mask_work, n_shells)
    spectrum_array = core.sf(core.rho_total(state, ctx), ctx)
    scale = float(core.fit_scale(spectrum_array, ctx, ctx.mask_work))
    spectrum = np.abs(np.asarray(spectrum_array))
    amp = np.asarray(ctx.amp)
    mult = np.asarray(ctx.mult)
    rows = []
    for name, mask in (("work", np.asarray(ctx.mask_work)), ("free", np.asarray(ctx.mask_free))):
        for shell in range(n_shells):
            here = (shells == shell) & mask
            if not here.any():
                continue
            model = scale * spectrum[here]
            observed = amp[here]
            weight = mult[here]
            r_value = float(np.sum(weight * np.abs(model - observed)) / np.sum(weight * observed))
            if model.std() > 0 and observed.std() > 0:
                cc = float(np.corrcoef(model, observed)[0, 1])
            else:
                cc = float("nan")
            rows.append({
                "set": name,
                "shell": int(shell),
                "d_min": float(1.0 / np.sqrt(np.max(s2[here]))),
                "n": int(here.sum()),
                "R": r_value,
                "CC": cc,
            })
    return rows


def point_row(ctx, meta, point, state):
    """Build the tidy row for one grid point.

    ``delta_r_free`` is the result; ``r_work`` is not, since a field with a million
    voxels against tens of thousands of work amplitudes can be driven almost
    anywhere. ``df_upper_bound`` and ``prior_shrinkage`` bracket the effective
    degrees of freedom from above and below -- the exact figure needs the Hessian
    spectrum, which costs thousands of matrix-vector products per point.

    Args:
        ctx: The prepared context.
        meta: Its metadata.
        point: The refine result for this grid point.
        state: Its refined state.

    Returns:
        A dict of one row's columns.
    """
    spectrum = core.r2c(core.delta_rho(state, ctx))
    power = ctx.mult * jnp.abs(spectrum) ** 2
    total = float(jnp.sum(power))
    outside = float(jnp.sum(jnp.where(ctx.mask_obs, 0.0, power)))
    return {
        "pdb_id": meta["pdb_id"],
        "zero_field": meta["zero_field"],
        "index": point["index"],
        "c": point["c"],
        "q_sigma": point["q_sigma"],
        "mu": point["mu"],
        "lr": point["lr"],
        "t_star": point["steps"],
        "stopped_on": point["stopped_on"],
        "epsilon_floor": point["epsilon_floor"],
        "epsilon_floor_estimator": point["epsilon_floor_estimator"],
        "e_work_final": point["e_work_final"],
        "e_work_over_floor": point["e_work_final"] / point["epsilon_floor"],
        "delta_r_free": float(core.delta_r_free(state, ctx)),
        "eta_c": float(core.eta_c(state, ctx)),
        "r_work": point["r_work_final"],
        "r_free": point["r_free_final"],
        "nullspace_energy_fraction": (outside / total) if total > 0 else 0.0,
        "df_upper_bound": float(jnp.sum(jnp.where(ctx.mask_work, ctx.mult, 0))),
        "prior_shrinkage": 1.0 - (1.0 - point["lr"]) ** point["steps"],
        "free_convention": meta["free_convention"],
        "free_fraction": meta["free_fraction"],
        "grid": "x".join(str(n) for n in meta["grid"]),
        "spacegroup": meta["spacegroup"],
    }


def log_to_wandb(rows, shells, meta, wandb_cfg):
    """Log one W&B run per grid point, if logging is enabled.

    ``wandb`` is imported here rather than at module scope, so ``mode: disabled``
    needs no account and no import. Every run carries the note that R_free is a trace
    only, which is meant to be read next to the curve.

    Args:
        rows: The tidy rows.
        shells: Per-shell metrics, keyed by grid index.
        meta: Dataset metadata, used for the run config.
        wandb_cfg: The ``wandb`` config group: mode, project, entity, group, job_type.

    Returns:
        True if runs were logged, False if logging was disabled.
    """
    mode = (wandb_cfg or {}).get("mode", "disabled")
    if mode == "disabled":
        return False
    import wandb

    for row in rows:
        run = wandb.init(
            project=wandb_cfg.get("project", "fieldrefine"),
            entity=wandb_cfg.get("entity"),
            group=wandb_cfg.get("group") or meta["pdb_id"],
            job_type=wandb_cfg.get("job_type", "grid"),
            mode=mode,
            notes=WANDB_NOTE,
            config={
                **{key: meta[key] for key in
                   ("pdb_id", "spacegroup", "d_min", "grid", "sample_rate", "free_convention",
                    "free_fraction", "sqrtS", "s_min", "zero_field")},
                **{key: row[key] for key in ("c", "q_sigma", "mu", "lr", "t_star",
                                             "epsilon_floor")},
            },
            reinit=True,
        )
        run.summary.update({
            key: value for key, value in row.items()
            if isinstance(value, (int, float, str, bool))
        })
        for shell in shells.get(row["index"], []):
            run.log({
                f"shell_{shell['set']}_R": shell["R"],
                f"shell_{shell['set']}_CC": shell["CC"],
                "shell_d_min": shell["d_min"],
            })
        run.finish()
    return True


def evaluate(pdb_id, root="data", zero_field=False, indices=None, wandb_cfg=None,
             n_shells=maps.N_SHELLS):
    """Evaluate every refined grid point into one tidy table.

    Points are read one at a time and dropped before the next, so memory does not
    grow with the size of the grid. Points that have not been refined are skipped.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.
        zero_field: Evaluate the zero-field grid.
        indices: Points to include; None considers the whole grid.
        wandb_cfg: The ``wandb`` config group, or None to disable logging.
        n_shells: Resolution shells for the per-shell metrics.

    Returns:
        The table as a ``pandas.DataFrame``, also written to CSV alongside a
        ``.shells.json`` of the per-shell metrics.

    Raises:
        EvaluateError: If no refined grid points were found.
    """
    ctx, meta = prepare.prepare(pdb_id, root, zero_field=zero_field)
    if indices is None:
        indices = range(refine.n_points())
    rows, shells = [], {}
    for index in indices:
        paths = refine.point_paths(pdb_id, index, root, zero_field)
        if not fetch.is_cached(paths["result"]) or not fetch.is_cached(paths["state"]):
            continue
        state, _, point = refine.read_point(paths)
        rows.append(point_row(ctx, meta, point, state))
        shells[int(index)] = shell_metrics(ctx, state, meta, n_shells)
        # one point at a time; drop it before the next
        del state
    if not rows:
        raise EvaluateError(
            f"no refined grid points found for {pdb_id}; run the refine stage first"
        )
    frame = pd.DataFrame(rows).sort_values("index")
    path = table_path(pdb_id, root, zero_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".shells.json").write_text(json.dumps(shells, indent=2, sort_keys=True))
    log_to_wandb(rows, shells, meta, wandb_cfg)
    return frame
