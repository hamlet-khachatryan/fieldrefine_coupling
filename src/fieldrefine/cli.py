"""Command line entry point, driven by Hydra.

Run from the repository root, which is where ``core.py``, ``device.py``, ``conf/`` and ``data/``
live::

    PYTHONPATH=src python -m fieldrefine.cli pdb_id=6LU7 stage=nullgate
    PYTHONPATH=src python -m fieldrefine.cli pdb_id=6LU7 zero_field=true stage=all
    PYTHONPATH=src python -m fieldrefine.cli pdb_id=6LU7 stage=refine grid.indices=[0,1,2]

Options and flags, all from ``conf/config.yaml`` and all overridable as ``key=value``:

``pdb_id``
    PDB entry code. Must have deposited structure factors **and** free flags; entries without a
    test set are rejected, and flags are never generated.
``stage``
    ``fetch``, ``prepare``, ``nullgate``, ``refine``, ``maps``, ``evaluate`` or ``all``. Stages are
    resumable: each reads the previous stage's cache rather than recomputing it.
``zero_field``
    ``true`` refines from nothing: rho0 is exactly zero, no model, no bulk-solvent mask, no scale
    fit. Caches are kept separate (``*_zero``), so a zero-field run never overwrites a model run.
``sample_rate``
    Grid points per d_min. Sets accuracy against gemmi's direct summation and the memory footprint;
    see the table in ``docs/02_prepare.md``.
``gate.*``
    ``n_steps``, ``q_sigma`` and ``seed`` for the null gate. The gate raises on failure and the
    pipeline stops there: its three thresholds are not negotiable.
``grid.*``
    ``indices`` (``null`` runs all 100 points), ``lr`` (``null`` estimates it from J alone),
    ``max_steps``, ``chunk``, ``seed``.
``maps.index``
    Which grid point to write maps for; ``null`` means the null point, c = 0 and mu = 0.
``wandb.mode``
    ``disabled`` (default, no account needed), ``offline`` or ``online``.

File formats read and written:

==============================  ==========================================================
``<ID>.cif`` / ``<ID>-sf.cif``  mmCIF coordinates and structure factors, from RCSB
``<ID>.mtz``                    reflections, converted by gemmi
``ctx[_zero].npz``              the ``core.Ctx`` arrays, minus the rebuilt symmetry arrays
``meta[_zero].json``            grid, symmetry, free-flag convention, fitted scales
``nullgate[_zero].json``        the three null numbers and their thresholds
``grid[_zero]/point_NNNN.*``    per-point state and trace, plus its knobs as JSON
``maps[_zero]/point_NNNN/``     CCP4 maps and one MTZ of coefficient sets, for Coot
``evaluate[_zero].csv``         one tidy row per grid point
==============================  ==========================================================

Worth knowing:

* Hydra does not change directory (``hydra.job.chdir: false``), so relative paths stay meaningful;
  run logs land under ``outputs/<pdb_id>/<timestamp>/``.
* ``maps`` and ``evaluate`` are imported lazily, so the early stages start without loading pandas
  or wandb, and a partial checkout still runs the gate.
* ``refine`` refuses to start unless the null gate passed for that mode.
* R_free is logged for the trace only. It is not an objective and must not be used to choose a
  stopping point, by optimiser or by eye.
"""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from . import fetch, nullgate, prepare, refine

log = logging.getLogger("fieldrefine")

STAGES = ("fetch", "prepare", "nullgate", "refine", "maps", "evaluate", "all")


def selected(stage, name):
    """Decide whether a stage should run.

    Args:
        stage: The configured stage, or ``"all"``.
        name: The stage being considered.

    Returns:
        True if this stage is the configured one, or if every stage was asked for.
    """
    return stage == "all" or stage == name


def resolve_indices(cfg):
    """Turn the configured grid selection into a list of point indices.

    Args:
        cfg: The Hydra configuration.

    Returns:
        None to run the whole grid, otherwise the selected indices. A bare integer
        is accepted so ``grid.indices=7`` and ``grid.indices=[7]`` behave alike.
    """
    indices = cfg.grid.indices
    if indices is None:
        return None
    if isinstance(indices, int):
        return [int(indices)]
    return [int(value) for value in indices]


def run(cfg):
    """Run the selected stages in pipeline order.

    Args:
        cfg: The Hydra configuration; see this module's docstring for the options.

    Returns:
        A summary dict with one key per stage that ran.

    Raises:
        SystemExit: If ``stage`` is not a recognised stage name.
        nullgate.NullGateError: If the null gate fails; the pipeline stops there.
        refine.RefineError: If the grid is asked for before the gate has passed.
    """
    stage = str(cfg.stage)
    if stage not in STAGES:
        raise SystemExit(f"stage must be one of {STAGES}, not {stage!r}")
    pdb_id, root, zero_field = str(cfg.pdb_id), str(cfg.root), bool(cfg.zero_field)
    summary = {"pdb_id": pdb_id, "zero_field": zero_field, "stage": stage}

    if selected(stage, "fetch"):
        info = fetch.fetch(pdb_id, root)
        log.info("fetched %s: %s, d_min %.2f, %d reflections",
                 pdb_id, info["spacegroup"], info["d_min"], info["n_reflections"])
        summary["fetch"] = {key: str(value) for key, value in info.items() if key != "dir"}

    if selected(stage, "prepare"):
        _, meta = prepare.prepare(pdb_id, root, sample_rate=float(cfg.sample_rate),
                                  zero_field=zero_field)
        log.info("prepared grid %s, %d work and %d free points, rho0 = %s",
                 meta["grid"], meta["n_work_grid"], meta["n_free_grid"], meta["rho0_source"])
        summary["prepare"] = {key: meta[key] for key in
                              ("grid", "spacegroup", "d_min", "n_work_grid", "n_free_grid",
                               "free_convention", "rho0_source", "sample_rate")}

    if selected(stage, "nullgate"):
        result = nullgate.gate(pdb_id, root, n_steps=int(cfg.gate.n_steps),
                               q_sigma=float(cfg.gate.q_sigma), zero_field=zero_field,
                               seed=int(cfg.gate.seed), raise_on_failure=True)
        log.info("null gate passed: dR_free %.3e, nullspace %.3e, symmetry %.3e",
                 result["delta_r_free"], result["nullspace_energy_fraction"],
                 result["symmetry_residual"])
        summary["nullgate"] = {key: result[key] for key in
                               ("delta_r_free", "nullspace_energy_fraction",
                                "symmetry_residual", "passed")}

    if selected(stage, "refine"):
        results = refine.refine(pdb_id, indices=resolve_indices(cfg), root=root,
                                zero_field=zero_field, lr=cfg.grid.lr,
                                max_steps=int(cfg.grid.max_steps), chunk=int(cfg.grid.chunk),
                                seed=int(cfg.grid.seed))
        log.info("refined %d grid points", len(results))
        summary["refine"] = {"n_points": len(results)}

    if selected(stage, "maps"):
        from . import maps

        written = maps.write_all(pdb_id, index=cfg.maps.index, root=root,
                                 zero_field=zero_field, write_mtz=bool(cfg.maps.write_mtz))
        log.info("wrote %d map files", len(written))
        summary["maps"] = [str(path) for path in written]

    if selected(stage, "evaluate"):
        from . import evaluate

        table = evaluate.evaluate(pdb_id, root=root, zero_field=zero_field,
                                  wandb_cfg=OmegaConf.to_container(cfg.wandb, resolve=True))
        log.info("evaluated %d grid points", len(table))
        summary["evaluate"] = {"n_rows": len(table)}

    return summary


# conf/ sits at the repository root, not inside the package, so resolve it from the
# working directory rather than from this file's location.
CONFIG_DIR = str((Path.cwd() / "conf").resolve())


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: log the resolved configuration, run, print the summary.

    Args:
        cfg: Composed from ``conf/config.yaml`` plus any command line overrides.
    """
    log.info("configuration:\n%s", OmegaConf.to_yaml(cfg))
    summary = run(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
