"""Run one precision-estimator configuration, addressed by a flat index.

Invoked as a module so ``job.sbatch`` can reach it::

    sbatch -C <class> job.sbatch fieldrefine.precision 6LU7 --index 0

There is **no sweep controller**. ``estimate.knob_grid`` is a pure index -> config map and
the SLURM array supplies the index, exactly as S10 requires; each task writes its own
file and nothing coordinates them.

R_free is computed for the readout and for nothing else. It enters neither the objective,
the stopping rule, nor the choice of any knob.
"""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp

import core

from .. import prepare
from . import estimate, gmrf


def _build(ctx, knobs, rf_seed, chunk):
    phi = gmrf.concat_features(
        gmrf.patch_features(ctx.rho0, knobs.patch, 1), whiten=True
    )
    return estimate.GCtx(
        **ctx._asdict(),
        inv_S=1.0 / (jnp.asarray(ctx.sqrtS) ** 2 + estimate.S_FLOOR),
        phi=phi,
        rf_seed=rf_seed,
        mu_graph=jnp.asarray(0.0),
        m=knobs.m,
        kernel=knobs.kernel,
        chunk=chunk,
        knn=None,
    )


def _mu_unit(g, ctx, key):
    """Return the mu at which the graph term matches the stationary term.

    A unit to sweep below, never a recommendation: on a real cell x'Lx exceeds
    x'P_stat x by about five orders, so mu at this value overwhelms the data term and
    R_work rises rather than falls.
    """
    x = core.pi_real(jax.random.normal(key, ctx.rho0.shape, dtype=jnp.float64), ctx)
    stat = float(jnp.sum(x * estimate.P_stat(x, g.inv_S)))
    graph = float(jnp.sum(x * estimate.P_graph(
        x, g.phi, g.rf_seed, g.m, 1.0, g.kernel, g.chunk
    )))
    return stat / graph, stat, graph


def main(argv=None):
    p = argparse.ArgumentParser(prog="fieldrefine.precision")
    p.add_argument("pdb_id")
    p.add_argument("--index", type=int, default=None,
                   help="knob grid index; defaults to SLURM_ARRAY_TASK_ID, then 0")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("--rf-seed", type=int, default=6107)
    p.add_argument("--eps-floor", type=float, default=0.0,
                   help="0 runs a fixed cap; a real floor comes from R_merge, never R_free")
    p.add_argument("--budget", type=float, default=8.0, help="GiB")
    p.add_argument("--root", default="data")
    args = p.parse_args(argv)

    index = args.index
    if index is None:
        index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    root = Path(args.root) / args.pdb_id.lower()
    ctx = prepare.read_ctx(
        str(root / "ctx.npz"), json.loads((root / "meta.json").read_text())
    )
    knobs = estimate.knob_grid(index, estimate.GRID)

    g = _build(ctx, knobs, args.rf_seed, args.chunk)
    n, d = int(g.phi.shape[0]), int(g.phi.shape[1])
    report = estimate.memory_report(n, d, knobs.m, args.chunk, int(args.budget * 2 ** 30))

    mu_bal, q_stat, q_graph = _mu_unit(g, ctx, jax.random.PRNGKey(args.rf_seed))
    mu = mu_bal * knobs.mu_scale
    gm = g._replace(mu_graph=jnp.asarray(mu))

    x, trace = estimate.solve(
        jnp.zeros(ctx.rho0.shape, dtype=jnp.float64), gm, args.eps_floor, args.steps
    )
    t = int(trace.t_star)
    row = {
        "index": index, "pdb_id": args.pdb_id, "kernel": knobs.kernel, "m": knobs.m,
        "patch": knobs.patch, "chunk": args.chunk, "rf_seed": args.rf_seed,
        "mu_scale": knobs.mu_scale, "mu": mu, "mu_bal": mu_bal,
        "x_p_stat_x": q_stat, "x_l_x": q_graph,
        "steps": args.steps, "t_star": t, "at_floor": bool(trace.at_floor),
        "r_work_start": float(trace.r_work[0]), "r_work_end": float(trace.r_work[t]),
        "delta_r_free": float(estimate.delta_r_free(x, gm)),
        "eta_c": float(estimate.eta_c(x, gm)),
        "peak_bytes": report.peak_bytes,
        "backend": jax.default_backend(),
    }
    # R_work rising means the prior beat the data; S8 calls that a solver problem
    row["converged"] = row["r_work_end"] <= row["r_work_start"]

    out = root / "precision"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"point_{index:04d}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps(row, indent=2, sort_keys=True))
    if not row["converged"]:
        print(
            f"WARNING: R_work rose {row['r_work_start']:.5f} -> {row['r_work_end']:.5f}; "
            f"mu={mu:.3e} overwhelms the data term. delta_r_free is not a measurement here."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
