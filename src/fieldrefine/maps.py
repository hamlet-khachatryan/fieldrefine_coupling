"""Stage 5: sigma_A weighted maps.

sigma_A, m and D are estimated **on work reflections only**, per resolution shell, for the model and
for the field solution separately. That departs from the usual practice of estimating on the test
set, and it is deliberate: the maps are for display, the free set is for one measurement.

``field_minus_model`` is a difference of two *coefficient sets* followed by a single inverse FFT,
never a subtraction of two real-space maps, so both sides are guaranteed to share a grid and a scale.
"""

import json
from pathlib import Path

import gemmi
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e, i1e

import core

from . import fetch, prepare, refine

N_SHELLS = 20
MIN_PER_SHELL = 50


class MapsError(RuntimeError):
    pass


def maps_dir(pdb_id, index, root="data", zero_field=False):
    """Give the output directory for one grid point's maps.

    Args:
        pdb_id: PDB entry code.
        index: Flat grid index.
        root: Cache root directory.
        zero_field: Select the zero-field output tree.

    Returns:
        The directory path; model and zero-field runs never share one.
    """
    directory = fetch.cache_paths(pdb_id, root)["dir"]
    return directory / ("maps_zero" if zero_field else "maps") / f"point_{index:04d}"


def shell_index(s2, mask, n_shells=N_SHELLS):
    """Assign every grid point to an equal-count resolution shell.

    Edges are quantiles of 1/d^2 over the masked points, so shells hold comparable
    numbers of reflections rather than comparable resolution ranges. The count is
    reduced when the data cannot support the requested number.

    Args:
        s2: 1/d^2 per grid point.
        mask: Points that define the quantiles; the work set.
        n_shells: Requested shell count.

    Returns:
        Tuple of the shell index per grid point and the shell count actually used.
    """
    values = np.asarray(s2)[np.asarray(mask)]
    if values.size < n_shells * MIN_PER_SHELL:
        n_shells = max(1, values.size // MIN_PER_SHELL)
    quantiles = np.linspace(0.0, 1.0, n_shells + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.digitize(np.asarray(s2), edges), n_shells


def centric_flags(spacegroup, shape):
    """Mark which reflections are centric.

    Centric and acentric reflections take different figure-of-merit expressions, so
    the distinction has to be made per reflection rather than per shell.

    Args:
        spacegroup: The ``gemmi.SpaceGroup``.
        shape: Real-space grid shape.

    Returns:
        A boolean array of shape ``h_shape(shape)``.
    """
    h0, h1, h2 = prepare.signed_miller(shape)
    hkl = np.stack([h0.ravel(), h1.ravel(), h2.ravel()], axis=1).astype(np.int32)
    flags = spacegroup.operations().centric_flag_array(hkl)
    return np.asarray(flags, dtype=bool).reshape(h0.shape)


def sigma_a_weights(amp, f_calc, mask_work, mask_obs, s2, centric, n_shells=N_SHELLS):
    """Per-shell D and sigma_A fitted on work; m and D returned on the whole H grid."""
    magnitude = np.abs(np.asarray(f_calc))
    amp = np.asarray(amp)
    shells, n_shells = shell_index(s2, mask_work, n_shells)
    work = np.asarray(mask_work)
    obs = np.asarray(mask_obs)
    d_field = np.zeros(amp.shape, dtype=np.float64)
    sigma_a_field = np.zeros(amp.shape, dtype=np.float64)
    e_obs = np.zeros(amp.shape, dtype=np.float64)
    e_calc = np.zeros(amp.shape, dtype=np.float64)
    table = []
    for shell in range(n_shells):
        in_shell = shells == shell
        fit = in_shell & work
        if not fit.any():
            continue
        obs_mean = float(np.mean(amp[fit] ** 2))
        calc_mean = float(np.mean(magnitude[fit] ** 2))
        cross = float(np.mean(amp[fit] * magnitude[fit]))
        d_value = cross / calc_mean if calc_mean > 0 else 0.0
        correlation = cross / np.sqrt(obs_mean * calc_mean) if obs_mean * calc_mean > 0 else 0.0
        sigma_a = float(np.clip(correlation, 0.0, 0.999))
        here = in_shell & obs
        d_field[here] = d_value
        sigma_a_field[here] = sigma_a
        if obs_mean > 0:
            e_obs[here] = amp[here] / np.sqrt(obs_mean)
        if calc_mean > 0:
            e_calc[here] = magnitude[here] / np.sqrt(calc_mean)
        table.append({
            "shell": shell,
            "d_min": float(1.0 / np.sqrt(np.max(np.asarray(s2)[here]))) if here.any() else None,
            "n_work": int(fit.sum()),
            "D": d_value,
            "sigma_a": sigma_a,
        })

    denominator = np.clip(1.0 - sigma_a_field ** 2, 1e-12, None)
    argument = 2.0 * sigma_a_field * e_obs * e_calc / denominator
    # i1e and i0e share the exp(-|x|) factor, so their ratio is the Bessel ratio
    m_acentric = np.asarray(i1e(jnp.asarray(argument)) / i0e(jnp.asarray(argument)))
    m_centric = np.tanh(np.clip(argument / 2.0, -50.0, 50.0))
    m_field = np.where(np.asarray(centric), m_centric, m_acentric)
    m_field = np.where(obs, np.clip(m_field, 0.0, 1.0), 0.0)
    return {"m": m_field, "D": d_field, "sigma_a": sigma_a_field, "shells": table}


def coefficient_sets(ctx, state, meta, n_shells=N_SHELLS):
    """Build every map coefficient set for one refined state.

    Produces 2fofc and fofc for the model and for the field, each with its own
    sigma_A weights, and ``field_minus_model`` as the **coefficient** difference of
    the two 2fofc sets -- never a subtraction of two real-space maps, so both sides
    are guaranteed to share a grid and a scale.

    In zero-field mode the model structure factors are identically zero and its phase
    is undefined, so the two model sets and the difference are omitted.

    Args:
        ctx: The prepared context.
        state: The refined state.
        meta: Its metadata, for the grid, cell and space group.
        n_shells: Resolution shells for the sigma_A estimation.

    Returns:
        Tuple of the coefficient sets keyed by name, the per-source weight tables,
        and the total density of the field solution.
    """
    shape = tuple(meta["grid"])
    spacegroup = gemmi.find_spacegroup_by_name(meta["spacegroup"])
    s2 = prepare.inverse_d2(gemmi.UnitCell(*meta["cell"]), shape)
    centric = centric_flags(spacegroup, shape)
    amp = np.asarray(ctx.amp)
    obs = np.asarray(ctx.mask_obs)

    f_model = np.asarray(core.sf(ctx.rho0, ctx))
    rho_field = core.rho_total(state, ctx)
    f_field = np.asarray(core.sf(rho_field, ctx))

    sets, weights = {}, {}
    for name, f_calc in (("model", f_model), ("field", f_field)):
        if not np.any(np.abs(f_calc) > 0):
            continue
        weight = sigma_a_weights(amp, f_calc, ctx.mask_work, ctx.mask_obs, s2, centric, n_shells)
        phase = np.zeros_like(f_calc)
        nonzero = np.abs(f_calc) > 0
        phase[nonzero] = f_calc[nonzero] / np.abs(f_calc[nonzero])
        sets[f"{name}_2fofc"] = np.where(
            obs, (2.0 * weight["m"] * amp - weight["D"] * np.abs(f_calc)) * phase, 0.0
        )
        sets[f"{name}_fofc"] = np.where(
            obs, (weight["m"] * amp - weight["D"] * np.abs(f_calc)) * phase, 0.0
        )
        weights[name] = weight
    if "field_2fofc" in sets and "model_2fofc" in sets:
        # one coefficient difference, one inverse FFT
        sets["field_minus_model"] = sets["field_2fofc"] - sets["model_2fofc"]
    return sets, weights, rho_field


def write_ccp4(values, meta, path):
    """Write one real-space map as CCP4.

    CCP4 is float32 by format; everything upstream is float64, and this is the only
    place the cast happens.

    Args:
        values: The real-space map.
        meta: Metadata carrying ``grid``, ``cell`` and ``spacegroup``.
        path: Destination path.

    Returns:
        The path written.
    """
    shape = tuple(meta["grid"])
    grid = gemmi.FloatGrid(*shape)
    grid.set_unit_cell(gemmi.UnitCell(*meta["cell"]))
    grid.spacegroup = gemmi.find_spacegroup_by_name(meta["spacegroup"])
    np.asarray(grid.array)[:] = np.asarray(values, dtype=np.float32)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ccp4.write_ccp4_map(str(path))
    return Path(path)


def write_mtz(sets, ctx, meta, path):
    """Write every coefficient set to one MTZ, for Coot.

    Values are sampled at the deposited Miller indices, conjugated back out of the
    half-grid fold, and written as an ``F_<set>`` / ``PHI_<set>`` pair per set, with
    phases in degrees.

    Args:
        sets: Coefficient sets keyed by name.
        ctx: The context, for ``mask_obs``.
        meta: Metadata carrying ``pdb_id``, ``grid``, ``cell`` and ``spacegroup``.
        path: Destination path.

    Returns:
        The path written.
    """
    shape = tuple(meta["grid"])
    mtz_in = fetch.read_mtz(fetch.cache_paths(meta["pdb_id"])["mtz"])
    hkl = mtz_in.make_miller_array().astype(np.int64)
    i0, i1, i2, conj = prepare.fold_indices(hkl[:, 0], hkl[:, 1], hkl[:, 2], shape)
    keep = np.asarray(ctx.mask_obs)[i0, i1, i2]
    hkl, i0, i1, i2, conj = hkl[keep], i0[keep], i1[keep], i2[keep], conj[keep]

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name(meta["spacegroup"])
    mtz.set_cell_for_all(gemmi.UnitCell(*meta["cell"]))
    mtz.add_dataset("fieldrefine")
    columns = [np.asarray(hkl[:, 0], dtype=np.float32),
               np.asarray(hkl[:, 1], dtype=np.float32),
               np.asarray(hkl[:, 2], dtype=np.float32)]
    for name, values in sets.items():
        picked = np.asarray(values)[i0, i1, i2]
        picked = np.where(conj, np.conj(picked), picked)
        mtz.add_column(f"F_{name}", "F")
        mtz.add_column(f"PHI_{name}", "P")
        columns.append(np.abs(picked).astype(np.float32))
        columns.append(np.degrees(np.angle(picked)).astype(np.float32))
    mtz.set_data(np.stack(columns, axis=1).astype(np.float32))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mtz.write_to_file(str(path))
    return Path(path)


def write_all(pdb_id, index=None, root="data", zero_field=False, write_mtz_file=True,
              n_shells=N_SHELLS, **kwargs):
    """Write every map for one grid point, plus the MTZ and a summary.

    Alongside the coefficient maps this writes Delta-rho and the sigma field
    directly, and the null-space component, which should be featureless noise at the
    1e-15 level -- structure there means the band limit was violated.

    Args:
        pdb_id: PDB entry code.
        index: Flat grid index; None selects the null point, c = 0 and mu = 0.
        root: Cache root directory.
        zero_field: Read the zero-field refinement.
        write_mtz_file: Also write the combined MTZ.
        n_shells: Resolution shells for the sigma_A estimation.
        **kwargs: Accepts ``write_mtz`` as an alias, for the config key of that name.

    Returns:
        The list of paths written.

    Raises:
        MapsError: If the grid point has not been refined, or on unknown arguments.
    """
    if "write_mtz" in kwargs:
        write_mtz_file = bool(kwargs.pop("write_mtz"))
    if kwargs:
        raise MapsError(f"unexpected arguments {sorted(kwargs)}")
    index = refine.null_index() if index is None else int(index)
    ctx, meta = prepare.prepare(pdb_id, root, zero_field=zero_field)
    paths = refine.point_paths(pdb_id, index, root, zero_field)
    if not fetch.is_cached(paths["state"]):
        raise MapsError(f"grid point {index} has not been refined: {paths['state']} is missing")
    state, _, point = refine.read_point(paths)

    sets, weights, _ = coefficient_sets(ctx, state, meta, n_shells)
    shape = tuple(meta["grid"])
    directory = maps_dir(pdb_id, index, root, zero_field)
    written = []
    for name, values in sets.items():
        written.append(write_ccp4(core.c2r(jnp.asarray(values), shape), meta,
                                  directory / f"{name}.ccp4"))
    drho = core.delta_rho(state, ctx)
    written.append(write_ccp4(drho, meta, directory / "delta_rho.ccp4"))
    written.append(write_ccp4(core.sigma_field(state.u, ctx), meta, directory / "sigma.ccp4"))
    outside = jnp.where(ctx.mask_obs, 0.0, core.r2c(drho))
    written.append(write_ccp4(core.c2r(outside, shape), meta, directory / "nullspace.ccp4"))
    if write_mtz_file and sets:
        written.append(write_mtz(sets, ctx, meta, directory / "coefficients.mtz"))

    summary = {
        "pdb_id": meta["pdb_id"],
        "index": index,
        "zero_field": bool(zero_field),
        "knobs": {key: point[key] for key in ("c", "q_sigma", "mu")},
        "sigma_a_estimated_on": "work reflections only",
        "coefficient_sets": sorted(sets),
        "shells": {name: weight["shells"] for name, weight in weights.items()},
        "files": [str(path) for path in written],
    }
    (directory / "maps.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return written
