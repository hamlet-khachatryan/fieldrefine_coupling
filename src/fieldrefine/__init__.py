"""FieldRefine pipeline: one PDB ID in, maps and coupling metrics out.

The package holds the six pipeline stages and the command line entry point. The
mathematical core (``core``) and the accelerator probe (``device``) stay flat at the
repository root, where SPEC.md requires them, and are imported from there.

Stages, in the order they run:

1. :mod:`fieldrefine.fetch` -- download and cache one PDB entry.
2. :mod:`fieldrefine.prepare` -- build a ``core.Ctx`` on one grid (the adapter).
3. :mod:`fieldrefine.nullgate` -- the mandatory null gate; nothing runs before it passes.
4. :mod:`fieldrefine.refine` -- the exhaustive knob grid.
5. :mod:`fieldrefine.maps` -- sigma_A weighted maps and one MTZ.
6. :mod:`fieldrefine.evaluate` -- one tidy row per grid point, to CSV and W&B.

:mod:`fieldrefine.cli` drives all six through Hydra; see ``conf/config.yaml``.

Run from the repository root, which is where ``core`` and ``device`` live::

    PYTHONPATH=src python -m fieldrefine.cli pdb_id=6LU7 stage=nullgate

The invariant that governs every stage: free reflections enter nothing except the
final delta_R_free readout. Not the data term, not the scale, not sigma_A, not the
stopping rule, not any hyperparameter.
"""

__all__ = ["fetch", "prepare", "nullgate", "refine", "maps", "evaluate", "cli"]
