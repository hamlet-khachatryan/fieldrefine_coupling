import json

import core

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fieldrefine import fetch, maps, nullgate, prepare

PDB_ID = "6LU7"
TOL_ANALYTIC = 1e-10
TOL_GEMMI = 1e-4  # rate 3 with REFMAC blur; the float32 floor of ~6e-6 needs rate >= 5
TOL_PROJECTOR = 1e-10
ACHIEVED = {}


@pytest.fixture(scope="module")
def dataset():
    info = fetch.fetch(PDB_ID)
    ctx, meta = prepare.prepare(PDB_ID)
    return {"info": info, "ctx": ctx, "meta": meta}


def _report(name, value):
    ACHIEVED[name] = value
    print(f"\nachieved[{name}] = {value:.3e}")


def test_01a_structure_factors_match_analytic_gaussian():
    cell = gemmi.UnitCell(30.0, 32.0, 34.0, 90.0, 90.0, 90.0)
    shape = (60, 64, 68)
    sigma = 1.0
    centre = np.array([0.2137, 0.4391, 0.6573])
    lengths = np.array([cell.a, cell.b, cell.c])
    spacing = lengths / np.array(shape)
    assert sigma / spacing.max() >= 2.0, (
        "fixture: the sampled Gaussian needs sigma/dx >= 2, otherwise its own aliasing "
        "dominates the comparison instead of the adapter (sigma/dx = 1.5 gives 6e-5)"
    )

    grids = np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij")
    delta = [(g - c + 0.5) % 1.0 - 0.5 for g, c in zip(grids, centre)]
    r2 = sum((d * length) ** 2 for d, length in zip(delta, lengths))
    rho = np.exp(-r2 / (2.0 * sigma ** 2)) / (2.0 * np.pi * sigma ** 2) ** 1.5
    assert rho.dtype == np.float64

    adapter = prepare.to_structure_factors(rho, cell)
    h0, h1, h2 = prepare.signed_miller(shape)
    s2 = prepare.inverse_d2(cell, shape)
    analytic = np.exp(
        2j * np.pi * (h0 * centre[0] + h1 * centre[1] + h2 * centre[2])
    ) * np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * s2)
    strong = np.abs(analytic) > 1e-3
    assert strong.sum() > 500, "fixture: not enough reflections above the comparison floor"
    error = np.abs(adapter[strong] - analytic[strong]) / np.abs(analytic[strong])
    _report("1a_analytic_gaussian_max_rel", float(error.max()))
    assert error.max() <= TOL_ANALYTIC, (
        f"adapter structure factors deviate from the analytic Gaussian transform by "
        f"{error.max():.3e}; convention is F_xtal = (V/sqrt(N)) * conj(r2c(rho))"
    )


def test_01b_structure_factors_match_gemmi(dataset):
    info, meta = dataset["info"], dataset["meta"]
    st = prepare.read_model(info["cif"])
    mtz = fetch.read_mtz(info["mtz"])
    shape = tuple(meta["grid"])
    rho_atoms, blur = prepare.density_map(
        st, meta["d_min"], shape, meta["sample_rate"], meta["density_cutoff"]
    )
    s2 = prepare.inverse_d2(st.cell, shape)
    adapter = prepare.to_structure_factors(rho_atoms, st.cell) * np.exp(blur * s2 / 4.0)

    ops = st.find_spacegroup().operations()
    hkl = mtz.make_miller_array()
    rng = np.random.default_rng(0)
    chosen = [h for h in hkl[rng.choice(len(hkl), 400, replace=False)]
              if not ops.is_systematically_absent([int(x) for x in h])][:150]
    calculator = gemmi.StructureFactorCalculatorX(st.cell)
    direct = np.array([calculator.calculate_sf_from_model(st[0], tuple(int(x) for x in h))
                       for h in chosen])
    keep = np.abs(direct) > 0.01 * np.abs(direct).max()
    chosen, direct = np.array(chosen)[keep], direct[keep]
    i0, i1, i2, conj = prepare.fold_indices(chosen[:, 0], chosen[:, 1], chosen[:, 2], shape)
    values = adapter[i0, i1, i2]
    values = np.where(conj, np.conj(values), values)
    error = np.abs(values - direct) / np.abs(direct)
    _report("1b_gemmi_max_rel", float(error.max()))
    assert error.max() <= TOL_GEMMI, (
        f"adapter disagrees with gemmi's direct summation by {error.max():.3e}; "
        "the floor is gemmi's float32 density grid"
    )


def test_02_project_sym_idempotent_for_real_spacegroup(dataset):
    ctx, meta = dataset["ctx"], dataset["meta"]
    idempotence, _ = prepare.projector_residuals(
        ctx.sym_idx, ctx.sym_phase, ctx.sym_conj, tuple(meta["grid"]), np.asarray(ctx.mult)
    )
    _report("02_projector_idempotence", idempotence)
    assert idempotence <= TOL_PROJECTOR


def test_03_project_sym_self_adjoint_for_real_spacegroup(dataset):
    ctx, meta = dataset["ctx"], dataset["meta"]
    _, self_adjointness = prepare.projector_residuals(
        ctx.sym_idx, ctx.sym_phase, ctx.sym_conj, tuple(meta["grid"]), np.asarray(ctx.mult)
    )
    _report("03_projector_self_adjointness", self_adjointness)
    assert self_adjointness <= TOL_PROJECTOR, (
        "self-adjointness is measured under the multiplicity-weighted inner product"
    )


def test_04_systematic_absences_present(dataset):
    ctx, meta = dataset["ctx"], dataset["meta"]
    shape = tuple(meta["grid"])
    ops = gemmi.find_spacegroup_by_name(meta["spacegroup"]).operations()
    rng = np.random.default_rng(1)
    symmetric = np.asarray(core.project_sym(
        core.r2c(jnp.asarray(rng.standard_normal(shape))),
        ctx.sym_idx, ctx.sym_phase, ctx.sym_conj,
    ))
    h0, h1, h2 = prepare.signed_miller(shape)
    flat = np.stack([h0.ravel(), h1.ravel(), h2.ravel()], axis=1)
    sample = flat[rng.choice(len(flat), 4000, replace=False)]
    absent = np.array([ops.is_systematically_absent([int(x) for x in h]) for h in sample])
    assert absent.sum() > 50, "fixture: this space group shows too few absences to test"
    i0, i1, i2, _ = prepare.fold_indices(sample[:, 0], sample[:, 1], sample[:, 2], shape)
    magnitude = np.abs(symmetric[i0, i1, i2])
    scale = magnitude.max()
    _report("04_absence_leakage", float(magnitude[absent].max() / scale))
    assert magnitude[absent].max() <= 1e-12 * scale, (
        "a symmetrised map must vanish where the space group forbids reflections"
    )
    assert magnitude[~absent].max() > 0.01 * scale


def test_05_mult_matches_friedel_structure(dataset):
    ctx, meta = dataset["ctx"], dataset["meta"]
    shape = tuple(meta["grid"])
    assert np.array_equal(np.asarray(ctx.mult), prepare.multiplicity(shape))
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal(shape))
    parseval = float(jnp.sum(ctx.mult * jnp.abs(core.r2c(x)) ** 2))
    direct = float(jnp.sum(x ** 2))
    _report("05_parseval_rel", abs(parseval - direct) / direct)
    assert abs(parseval - direct) <= 1e-10 * direct


def test_06_mask_identity_on_real_flags(dataset):
    ctx, meta = dataset["ctx"], dataset["meta"]
    work = np.asarray(ctx.mask_work)
    free = np.asarray(ctx.mask_free)
    assert np.array_equal(np.asarray(ctx.mask_obs), work | free)
    assert not np.any(work & free)
    assert meta["n_work_grid"] > 0 and meta["n_free_grid"] > 0
    fraction = meta["n_free_grid"] / (meta["n_work_grid"] + meta["n_free_grid"])
    low, high = prepare.FREE_FRACTION_RANGE
    assert low <= fraction <= high, f"free fraction {fraction:.3f} is implausible"


def test_07_scale_fit_ignores_free_reflections(dataset):
    info, meta = dataset["info"], dataset["meta"]
    st = prepare.read_model(info["cif"])
    mtz = fetch.read_mtz(info["mtz"])
    shape = tuple(meta["grid"])
    rho_atoms, blur = prepare.density_map(
        st, meta["d_min"], shape, meta["sample_rate"], meta["density_cutoff"]
    )
    s2 = prepare.inverse_d2(st.cell, shape)
    f_calc = prepare.to_structure_factors(rho_atoms, st.cell) * np.exp(blur * s2 / 4.0)
    f_mask = prepare.to_structure_factors(prepare.solvent_mask_map(st, shape), st.cell)
    free_value, _, _ = prepare.free_flag_convention(
        {int(k): v for k, v in meta["free_flag_counts"].items()}
    )
    labels = {"f": info["f_label"], "sig": info["sig_label"], "free": info["free_label"]}
    data = prepare.reciprocal_arrays(mtz, shape, labels, free_value, st.find_spacegroup())

    baseline = prepare.fit_scales(
        f_calc, f_mask, data["amp"], data["wgt"], data["mask_work"], s2, shape
    )
    perturbed_amp = np.where(data["mask_free"], data["amp"] * 1.5 + 10.0, data["amp"])
    perturbed = prepare.fit_scales(
        f_calc, f_mask, perturbed_amp, data["wgt"], data["mask_work"], s2, shape
    )
    assert baseline["k_sol"] == perturbed["k_sol"]
    assert baseline["b_sol"] == perturbed["b_sol"]
    difference = float(np.max(np.abs(baseline["coefficients"] - perturbed["coefficients"])))
    _report("07_scale_shift_on_free_perturbation", difference)
    assert difference == 0.0, (
        "the scale fit moved when free-set amplitudes changed; it must use mask_work only"
    )


def test_08_one_grid_shape_and_no_single_precision(dataset):
    ctx, meta = dataset["ctx"], dataset["meta"]
    shape = tuple(meta["grid"])
    h = prepare.h_shape(shape)
    assert ctx.rho0.shape == shape
    for name in ("amp", "wgt", "mult", "mask_work", "mask_free", "mask_obs",
                 "sqrtS", "sqrtS_sigma", "F0"):
        assert getattr(ctx, name).shape == h, name
    for name in ("sym_idx", "sym_phase", "sym_conj"):
        assert getattr(ctx, name).shape == (meta["n_operators"],) + h, name

    state = core.State(jnp.zeros(shape, dtype=jnp.float64), jnp.zeros(shape, dtype=jnp.float64))
    assert core.delta_rho(state, ctx).shape == shape
    assert core.sigma_field(state.u, ctx).shape == shape
    assert core.rho_total(state, ctx).shape == shape
    assert prepare.assert_float64(ctx)
    for name, value in zip(core.Ctx._fields, ctx):
        dtype = np.asarray(value).dtype
        assert dtype not in (np.dtype(np.float32), np.dtype(np.complex64)), name
    assert core.delta_rho(state, ctx).dtype == jnp.float64
    assert core.sf(ctx.rho0, ctx).dtype == jnp.complex128


def test_09_metadata_records_the_choices(dataset):
    meta = dataset["meta"]
    assert meta["scale_fitted_on"] == "mask_work only"
    assert meta["sigma_a_estimated_on"] == "work reflections only"
    assert meta["sqrtS"] == "flat"
    assert meta["free_convention"]
    low, high = prepare.FREE_FRACTION_RANGE
    assert low <= meta["free_fraction"] <= high
    fractions = meta["free_flag_fractions"]
    assert fractions
    # metadata stores fractions to 6 decimals; exact counts live in free_flag_counts
    assert abs(sum(fractions.values()) - 1.0) <= 5e-7 * len(fractions)
    counts = meta["free_flag_counts"]
    assert set(counts) == set(fractions)
    total = sum(counts.values())
    assert abs(counts[str(meta["free_value"])] / total - meta["free_fraction"]) < 1e-9
    assert meta["projector_idempotence"] <= TOL_PROJECTOR
    assert meta["projector_self_adjointness"] <= TOL_PROJECTOR
    paths = prepare.ctx_paths(PDB_ID)
    assert json.loads(paths["meta"].read_text())["pdb_id"] == PDB_ID


def _synthetic_ctx(seed, shape=(16, 16, 16)):
    return core.make_ctx(jax.random.PRNGKey(seed), shape)


def test_10_null_gate_passes_on_synthetic_data():
    result = nullgate.run_gate(_synthetic_ctx(10), n_steps=50)
    for name in ("delta_r_free", "nullspace_energy_fraction", "symmetry_residual"):
        _report(f"10_{name}", abs(result[name]))
    assert result["delta_rho_norm"] > 0.0, "fixture: the refinement must move something"
    assert result["r_work_end"] < result["r_work_start"], "fixture: R_work must fall"
    assert result["passed"], result["checks"]


def test_11_null_gate_fails_when_free_reflections_enter_the_work_set():
    ctx = _synthetic_ctx(11)
    leaking = ctx._replace(mask_work=ctx.mask_obs)
    result = nullgate.run_gate(leaking, n_steps=50)
    _report("11_delta_r_free_when_leaking", abs(result["delta_r_free"]))
    assert not result["passed"], "the gate did not fire on a deliberately leaking work set"
    assert not result["checks"]["delta_r_free"]
    with pytest.raises(nullgate.NullGateError) as raised:
        nullgate.assert_gate(result, "synthetic-leaking")
    message = str(raised.value)
    assert "mask_obs rather than mask_work" in message
    assert "do not loosen the thresholds" in message


def test_13_field_minus_model_is_a_coefficient_difference():
    shape = (16, 18, 20)
    rng = np.random.default_rng(13)
    first = core.r2c(jnp.asarray(rng.standard_normal(shape)))
    second = core.r2c(jnp.asarray(rng.standard_normal(shape)))
    by_coefficients = core.c2r(first - second, shape)
    by_maps = core.c2r(first, shape) - core.c2r(second, shape)
    scale = float(jnp.max(jnp.abs(by_coefficients)))
    error = float(jnp.max(jnp.abs(by_coefficients - by_maps))) / scale
    _report("13_field_minus_model_rel", error)
    assert error <= 1e-10, (
        "field_minus_model must be a coefficient difference with one inverse FFT"
    )


def test_14_maps_round_trip_through_ccp4(tmp_path):
    shape = (16, 18, 20)
    meta = {"grid": list(shape), "cell": [30.0, 32.0, 34.0, 90.0, 90.0, 90.0], "spacegroup": "P 1"}
    rng = np.random.default_rng(14)
    values = rng.standard_normal(shape)
    path = maps.write_ccp4(values, meta, tmp_path / "round_trip.ccp4")
    back = gemmi.read_ccp4_map(str(path))
    recovered = np.asarray(back.grid.array, dtype=np.float64)
    assert recovered.shape == shape
    # CCP4 is float32 by format; everything upstream is float64
    error = float(np.max(np.abs(recovered - values)) / np.max(np.abs(values)))
    _report("14_ccp4_round_trip_rel", error)
    assert error <= 1e-6


def test_15_zero_field_gate_passes_and_reduces_r_work():
    ctx = _synthetic_ctx(15)
    zero = ctx._replace(rho0=jnp.zeros_like(ctx.rho0), F0=jnp.zeros_like(ctx.F0))
    state = nullgate.start_state(nullgate.null_ctx(zero), seed=15)
    assert float(jnp.max(jnp.abs(state.z))) > 0.0, "zero field must seed a non-zero start"
    seeded = np.abs(np.asarray(core.r2c(state.z)))
    assert seeded[np.asarray(ctx.mask_free)].max() <= 1e-12 * seeded.max(), (
        "the seeded start must carry no free-set information"
    )
    result = nullgate.run_gate(zero, n_steps=50, seed=15)
    for name in ("delta_r_free", "nullspace_energy_fraction", "symmetry_residual"):
        _report(f"15_{name}", abs(result[name]))
    assert result["passed"], result["checks"]
    assert result["r_work_end"] < result["r_work_start"], (
        f"zero-field R_work rose, {result['r_work_start']:.4f} -> {result['r_work_end']:.4f}; "
        "the seeded start is mis-scaled if the prior swamps the data term"
    )


def test_12_symmetry_arrays_are_rebuilt_by_the_loader(dataset):
    meta = dataset["meta"]
    paths = prepare.ctx_paths(PDB_ID)
    with np.load(paths["ctx"]) as data:
        stored = set(data.files)
    for name in prepare.DERIVED_FIELDS:
        assert name not in stored, f"{name} must be rebuilt by the loader, not stored"
    assert {"rho0", "mult", "amp"} <= stored
    _report("12_ctx_npz_MB", paths["ctx"].stat().st_size / 1e6)

    ctx = prepare.read_ctx(paths["ctx"], meta)
    shape = tuple(meta["grid"])
    spacegroup = gemmi.find_spacegroup_by_name(meta["spacegroup"])
    sym_idx, sym_phase, sym_conj = prepare.symmetry_arrays(spacegroup, shape)
    assert np.array_equal(np.asarray(ctx.sym_idx), np.asarray(sym_idx))
    assert np.array_equal(np.asarray(ctx.sym_conj), np.asarray(sym_conj))
    assert float(jnp.max(jnp.abs(ctx.sym_phase - sym_phase))) == 0.0
    idempotence, self_adjointness = prepare.projector_residuals(
        ctx.sym_idx, ctx.sym_phase, ctx.sym_conj, shape, np.asarray(ctx.mult)
    )
    assert idempotence <= TOL_PROJECTOR and self_adjointness <= TOL_PROJECTOR
