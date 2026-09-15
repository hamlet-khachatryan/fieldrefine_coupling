import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import core

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
SHAPE = (8, 10, 12)
ODD_SHAPE = (7, 5, 9)
Q_SIGMA = 2.0
N_NULL = 50

TOL = 1e-10
TOL_NULL = 1e-12
# measured FFT round-trip leakage is ~5e-16
TOL_ROUNDTRIP = 1e-13
TOL_FD = 1e-5
TOL_CALIBRATION = 0.05
NONZERO = 1e-6

ROUNDTRIP_MSG = (
    "exact zero is unattainable in float64: r2c(c2r(G)) leaks ~5e-16 of max|G| "
    "into coefficients where G is exactly 0; achievable bound is 1e-13 relative"
)


def _key(seed):
    return jax.random.PRNGKey(seed)


def _h(shape):
    return (shape[0], shape[1], shape[2] // 2 + 1)


def _f64(v):
    return jnp.asarray(v, dtype=jnp.float64)


def _normal(seed, shape):
    return jax.random.normal(_key(seed), shape, dtype=jnp.float64)


def _cnormal(seed, shape):
    k1, k2 = jax.random.split(_key(seed))
    re_ = jax.random.normal(k1, shape, dtype=jnp.float64)
    im_ = jax.random.normal(k2, shape, dtype=jnp.float64)
    return re_ + 1j * im_


def _random_state(seed, shape=SHAPE):
    k1, k2 = jax.random.split(_key(seed))
    return core.State(
        jax.random.normal(k1, shape, dtype=jnp.float64),
        jax.random.normal(k2, shape, dtype=jnp.float64),
    )


def _copy(tree):
    return jax.tree_util.tree_map(lambda a: jnp.array(a, copy=True), tree)


def _rel(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    scale = float(jnp.maximum(jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))))
    diff = float(jnp.max(jnp.abs(a - b)))
    return diff if scale == 0.0 else diff / scale


def _conj_planes(shape):
    return [0] + ([shape[2] // 2] if shape[2] % 2 == 0 else [])


def _mult_ref(shape):
    m = np.full(_h(shape), 2, dtype=np.int64)
    for p in _conj_planes(shape):
        m[:, :, p] = 1
    return m


def _reflect(a, shape):
    a = np.asarray(a)
    out = a.copy()
    i = (-np.arange(shape[0])) % shape[0]
    j = (-np.arange(shape[1])) % shape[1]
    for p in _conj_planes(shape):
        out[:, :, p] = a[i][:, j, p]
    return out


def _friedel_sym(a, shape):
    a = np.asarray(a, dtype=np.float64)
    return jnp.asarray(0.5 * (a + _reflect(a, shape)))


def _ip_real(x, y):
    return float(jnp.sum(x * y))


def _ip_mult(F, G, mult):
    return float(jnp.sum(mult * jnp.real(F * jnp.conj(G))))


def _index_radius(shape):
    q0 = np.fft.fftfreq(shape[0]) * shape[0]
    q1 = np.fft.fftfreq(shape[1]) * shape[1]
    q2 = np.fft.rfftfreq(shape[2]) * shape[2]
    return np.sqrt(q0[:, None, None] ** 2 + q1[None, :, None] ** 2 + q2[None, None, :] ** 2)


def _twofold_sym(shape, t0=1):
    H = _h(shape)
    h0 = np.arange(H[0])[:, None, None]
    h1 = np.arange(H[1])[None, :, None]
    h2 = np.arange(H[2])[None, None, :]
    # x(-r0+t0,-r1,r2) -> F(-h0,-h1,h2) exp(-2pi*i*h0*t0/n0)
    src = np.broadcast_to(((-h0) % H[0]) * H[1] * H[2] + ((-h1) % H[1]) * H[2] + h2, H)
    ident = np.arange(np.prod(H)).reshape(H)
    phase = np.broadcast_to(np.exp(-2j * np.pi * h0 * t0 / shape[0]), H)
    sym_idx = np.stack([ident, src]).astype(np.int32)
    sym_phase = np.stack([np.ones(H, dtype=np.complex128), phase]).astype(np.complex128)
    return jnp.asarray(sym_idx), jnp.asarray(sym_phase)


def _apply_op(G, sym_idx, sym_phase, k):
    return G.ravel()[sym_idx[k]] * sym_phase[k]


def _sym_ctx(seed, shape=SHAPE):
    ctx = core.make_ctx(_key(seed), shape)
    sym_idx, sym_phase = _twofold_sym(shape)
    work = ctx.mask_work & ctx.mask_work.ravel()[sym_idx[1]]
    free = ctx.mask_free & ctx.mask_free.ravel()[sym_idx[1]]
    return ctx._replace(
        sym_idx=sym_idx, sym_phase=sym_phase,
        mask_work=work, mask_free=free, mask_obs=work | free,
    )


def _full_ctx(seed, state, c=0.5, mu=1.0):
    shape = state.z.shape
    ctx = _sym_ctx(seed, shape)
    ctx = ctx._replace(sqrtS_sigma=core.sigma_spectrum(c, Q_SIGMA, ctx), mu=_f64(mu))
    rho = np.asarray(core.rho_total(state, ctx))
    ctx = ctx._replace(rho_floor=_f64(np.quantile(rho, 0.25)))
    active = float(np.mean(np.asarray(core.neg_part(core.rho_total(state, ctx), ctx)) < 0.0))
    assert 0.1 < active < 0.9, f"fixture: e_pos must be active on part of the grid, got {active}"
    return ctx


def _null_ctx(seed, shape=SHAPE, c=0.0, mu=0.0):
    ctx = core.make_ctx(_key(seed), shape)
    sym_idx, sym_phase = core.identity_sym(_h(shape))
    ctx = ctx._replace(sym_idx=sym_idx, sym_phase=sym_phase, mu=_f64(mu))
    return ctx._replace(sqrtS_sigma=core.sigma_spectrum(c, Q_SIGMA, ctx))


def _assert_null_masks(ctx, shape):
    work = np.asarray(ctx.mask_work)
    free = np.asarray(ctx.mask_free)
    obs = np.asarray(ctx.mask_obs)
    assert work.any() and free.any(), "fixture: mask_work and mask_free must be non-empty"
    assert not (work & free).any(), "fixture: mask_work and mask_free must be disjoint"
    assert np.array_equal(obs, work | free), "fixture: mask_obs != mask_work | mask_free"
    assert np.array_equal(work, _reflect(work, shape)), "fixture: mask_work not Friedel-symmetric"
    assert np.array_equal(free, _reflect(free, shape)), "fixture: mask_free not Friedel-symmetric"
    assert ctx.sym_idx.shape[0] == 1, "fixture: null configuration uses identity_sym"


def _assert_constant_sigma(ctx, shape):
    assert np.array_equal(np.asarray(ctx.sqrtS_sigma), np.zeros(_h(shape))), (
        "c == 0 must give sqrtS_sigma exactly 0"
    )
    sigma = np.asarray(core.sigma_field(_normal(9999, shape), ctx))
    assert np.all(sigma == sigma.flat[0]), "c == 0 must give exactly constant sigma"


def _null_start(seed, shape=SHAPE):
    return core.State(jnp.zeros(shape, dtype=jnp.float64), _normal(seed, shape))


def _lr(state, ctx, seed):
    def jfun(s):
        return core.objective(s, ctx)[0]

    hvp = jax.jit(lambda v: jax.jvp(jax.grad(jfun), (state,), (v,))[1])
    v = _random_state(seed, state.z.shape)
    lam = 0.0
    for _ in range(50):
        norm = float(jnp.sqrt(jnp.sum(v.z ** 2) + jnp.sum(v.u ** 2)))
        v = core.State(v.z / norm, v.u / norm)
        w = hvp(v)
        lam = float(jnp.sqrt(jnp.sum(w.z ** 2) + jnp.sum(w.u ** 2)))
        v = w
    assert np.isfinite(lam) and lam > 0.0, f"fixture: Hessian norm estimate {lam}"
    return 0.5 / lam


def _refine(ctx, seed, shape=SHAPE, n=N_NULL, lr=None):
    state0 = _null_start(seed, shape)
    if lr is None:
        lr = _lr(state0, ctx, seed + 1)
    final, trace = core.run(_copy(state0), ctx, lr, n)
    return state0, final, trace


def _band_max(x, mask):
    return float(jnp.max(jnp.where(mask, jnp.abs(core.r2c(x)), 0.0)))


def _r_fitted_on_work(F, ctx, mask):
    k = core.fit_scale(F, ctx, ctx.mask_work)
    return float(core.r_factor(F, ctx, k, mask))


def _F_of(state, ctx):
    return core.sf(core.rho_total(state, ctx), ctx)


_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
gpu = pytest.mark.skipif(not _HAS_GPU, reason="no GPU visible to JAX")


def test_01_x64_active_computed_array_is_float64():
    assert jax.config.read("jax_enable_x64"), "importing core must enable jax_enable_x64"
    x = jax.random.normal(_key(1), SHAPE)
    assert x.dtype == jnp.float64
    assert core.c2r(core.r2c(x), SHAPE).dtype == jnp.float64


def test_02_computed_spectrum_is_complex128():
    F = core.r2c(_normal(2, SHAPE))
    assert F.dtype == jnp.complex128
    assert F.shape == _h(SHAPE)


def test_03_jit_and_nonjit_objective_agree():
    state = _random_state(3)
    ctx = _full_ctx(3, state)
    J_jit, terms_jit = jax.jit(core.objective)(state, ctx)
    with jax.disable_jit():
        J_eager, terms_eager = core.objective(state, ctx)
    assert _rel(J_jit, J_eager) <= TOL
    for name, a, b in zip(terms_jit._fields, terms_jit, terms_eager):
        assert _rel(a, b) <= TOL, name


def test_04_scan_run_matches_python_loop_over_step():
    n = 7
    state = _random_state(4)
    ctx = _full_ctx(4, state)
    lr = _lr(state, ctx, 40)
    final, trace = core.run(_copy(state), ctx, lr, n)
    s = _copy(state)
    j_pre, j_post, e_pre, e_post = [], [], [], []
    for _ in range(n):
        J, terms = core.objective(s, ctx)
        j_pre.append(float(J))
        e_pre.append(float(terms.e_work))
        s = core.step(s, ctx, lr)
        J, terms = core.objective(s, ctx)
        j_post.append(float(J))
        e_post.append(float(terms.e_work))
    assert _rel(final.z, s.z) <= TOL
    assert _rel(final.u, s.u) <= TOL
    for name, col in zip(trace._fields, trace):
        assert col.shape == (n,), name
        assert bool(jnp.all(jnp.isfinite(col))), name
    aligned = [
        _rel(trace.j, jnp.asarray(j)) <= TOL and _rel(trace.e_work, jnp.asarray(e)) <= TOL
        for j, e in ((j_pre, e_pre), (j_post, e_post))
    ]
    assert any(aligned), "trace rows must equal objective at every pre-step or every post-step state"


def test_05_no_third_party_module_beyond_jax_numpy_pytest():
    allowed = {"jax", "numpy", "pytest"}
    roots = set()
    for node in ast.walk(ast.parse((ROOT / "core.py").read_text())):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "core.py must not use relative imports"
            roots.add(node.module.split(".")[0])
    bad = {r for r in roots if r not in sys.stdlib_module_names and r not in allowed}
    assert not bad, f"core.py imports third-party modules {bad}"
    probe = (
        "import sys\n"
        "import jax, jax.numpy as jnp, numpy\n"
        "jnp.fft.rfftn(jnp.ones((4, 4, 4)), norm='ortho')\n"
        "before = {m.split('.')[0] for m in sys.modules}\n"
        "import core\n"
        "core.c2r(core.r2c(jnp.ones((4, 4, 4))), (4, 4, 4))\n"
        "after = {m.split('.')[0] for m in sys.modules}\n"
        "print(' '.join(sorted(after - before)))\n"
    )
    env = dict(os.environ, JAX_ENABLE_X64="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    added = set(out.stdout.split())
    extra = {m for m in added if m not in sys.stdlib_module_names and m not in allowed | {"core"}}
    assert not extra, (
        f"importing core loaded third-party modules {extra}; "
        "modules jax itself loads are attributed to jax"
    )


def test_06_parseval_under_multiplicity_inner_product():
    for seed, shape in ((6, SHAPE), (60, ODD_SHAPE)):
        ctx = core.make_ctx(_key(seed), shape)
        assert ctx.mult.shape == _h(shape)
        assert jnp.issubdtype(ctx.mult.dtype, jnp.integer), "mult must be an integer array"
        assert np.array_equal(np.asarray(ctx.mult), _mult_ref(shape)), (
            "mult must be 1 on the h2=0 and Nyquist planes, 2 elsewhere"
        )
        x = _normal(seed + 1, shape)
        F = core.r2c(x)
        assert _rel(jnp.sum(x ** 2), jnp.sum(ctx.mult * jnp.abs(F) ** 2)) <= TOL, shape


def test_07_c2r_r2c_roundtrip_is_identity():
    for seed, shape in ((7, SHAPE), (70, ODD_SHAPE)):
        x = _normal(seed, shape)
        y = core.c2r(core.r2c(x), shape)
        assert y.shape == shape
        assert _rel(y, x) <= TOL, shape


def test_08_r2c_c2r_adjoint_under_multiplicity_inner_product():
    for seed, shape in ((8, SHAPE), (80, ODD_SHAPE)):
        mult = jnp.asarray(_mult_ref(shape))
        x = _normal(seed, shape)
        y = _normal(seed + 1, shape)
        w = _friedel_sym(np.abs(np.asarray(_normal(seed + 2, _h(shape)))), shape)
        for G in (core.r2c(y), core.r2c(y) * w):
            lhs = _ip_mult(core.r2c(x), G, mult)
            rhs = _ip_real(x, core.c2r(G, shape))
            assert abs(lhs - rhs) <= TOL * max(abs(lhs), abs(rhs)), (
                f"<r2c(x),G>_mult = {lhs} but <x,c2r(G)> = {rhs}; adjoint constant must be 1"
            )


def test_09_apply_mult_identity_and_self_adjoint():
    x = _normal(9, SHAPE)
    y = _normal(90, SHAPE)
    ones = jnp.ones(_h(SHAPE), dtype=jnp.float64)
    assert _rel(core.apply_mult(x, ones), x) <= TOL
    s = _friedel_sym(_normal(91, _h(SHAPE)), SHAPE)
    lhs = _ip_real(core.apply_mult(x, s), y)
    rhs = _ip_real(x, core.apply_mult(y, s))
    assert abs(lhs - rhs) <= TOL * max(abs(lhs), abs(rhs))


def test_10_apply_mult_composes_multiplicatively():
    x = _normal(10, SHAPE)
    s = _friedel_sym(_normal(100, _h(SHAPE)), SHAPE)
    t = _friedel_sym(_normal(101, _h(SHAPE)), SHAPE)
    lhs = core.apply_mult(core.apply_mult(x, s), t)
    rhs = core.apply_mult(x, s * t)
    assert _rel(lhs, rhs) <= TOL


def test_11_project_band_idempotent():
    ctx = core.make_ctx(_key(11), SHAPE)
    F = _cnormal(110, _h(SHAPE))
    P = core.project_band(F, ctx.mask_obs)
    assert np.array_equal(np.asarray(core.project_band(P, ctx.mask_obs)), np.asarray(P))


def test_12_project_band_self_adjoint():
    ctx = core.make_ctx(_key(12), SHAPE)
    mult = jnp.asarray(_mult_ref(SHAPE))
    F = _cnormal(120, _h(SHAPE))
    G = _cnormal(121, _h(SHAPE))
    lhs = _ip_mult(core.project_band(F, ctx.mask_obs), G, mult)
    rhs = _ip_mult(F, core.project_band(G, ctx.mask_obs), mult)
    assert abs(lhs - rhs) <= TOL * max(abs(lhs), abs(rhs))


def test_13_project_sym_idempotent():
    sym_idx, sym_phase = _twofold_sym(SHAPE)
    F = _cnormal(13, _h(SHAPE))
    TT = _apply_op(_apply_op(F, sym_idx, sym_phase, 1), sym_idx, sym_phase, 1)
    assert _rel(TT, F) <= TOL, "fixture: the two-operator set must close into a group"
    P = core.project_sym(F, sym_idx, sym_phase)
    assert _rel(P, F) > 0.1, "fixture: projection must be non-trivial"
    assert _rel(core.project_sym(P, sym_idx, sym_phase), P) <= TOL


def test_14_project_sym_self_adjoint():
    sym_idx, sym_phase = _twofold_sym(SHAPE)
    mult = jnp.asarray(_mult_ref(SHAPE))
    F = _cnormal(14, _h(SHAPE))
    G = _cnormal(140, _h(SHAPE))
    lhs = _ip_mult(core.project_sym(F, sym_idx, sym_phase), G, mult)
    rhs = _ip_mult(F, core.project_sym(G, sym_idx, sym_phase), mult)
    assert abs(lhs - rhs) <= TOL * max(abs(lhs), abs(rhs))


def test_15_pi_real_idempotent():
    ctx = _sym_ctx(15)
    x = _normal(150, SHAPE)
    p = core.pi_real(x, ctx)
    assert _rel(p, x) > 0.1, "fixture: projection must be non-trivial"
    assert _rel(core.pi_real(p, ctx), p) <= TOL


def test_16_pi_real_self_adjoint():
    ctx = _sym_ctx(16)
    x = _normal(160, SHAPE)
    y = _normal(161, SHAPE)
    lhs = _ip_real(core.pi_real(x, ctx), y)
    rhs = _ip_real(x, core.pi_real(y, ctx))
    assert abs(lhs - rhs) <= TOL * max(abs(lhs), abs(rhs))


def test_17_delta_rho_band_limited_to_mask_obs():
    state = _random_state(17)
    ctx = _full_ctx(17, state)
    assert bool(jnp.any(~ctx.mask_obs)), "fixture: mask_obs must exclude some coefficients"
    drho = core.delta_rho(state, ctx)
    inside = _band_max(drho, ctx.mask_obs)
    outside = _band_max(drho, ~ctx.mask_obs)
    assert inside > 0.0
    assert outside <= TOL_ROUNDTRIP * inside, f"{outside / inside:.3e}; {ROUNDTRIP_MSG}"


def test_18_delta_rho_band_limited_after_20_steps():
    state = _random_state(18)
    ctx = _full_ctx(18, state)
    lr = _lr(state, ctx, 180)
    final, _ = core.run(_copy(state), ctx, lr, 20)
    drho = core.delta_rho(final, ctx)
    inside = _band_max(drho, ctx.mask_obs)
    outside = _band_max(drho, ~ctx.mask_obs)
    assert inside > 0.0
    assert outside <= TOL_ROUNDTRIP * inside, f"{outside / inside:.3e}; {ROUNDTRIP_MSG}"


def test_19_delta_rho_real_and_finite():
    state = _random_state(19)
    state = core.State(3.0 * state.z, 3.0 * state.u)
    ctx = _full_ctx(19, state)
    drho = core.delta_rho(state, ctx)
    assert drho.shape == SHAPE
    assert drho.dtype == jnp.float64
    assert not jnp.iscomplexobj(drho)
    assert bool(jnp.all(jnp.isfinite(drho)))


def test_20_sigma_field_bounded_below_by_s_min():
    ctx = core.make_ctx(_key(20), SHAPE)
    ctx = ctx._replace(sqrtS_sigma=core.sigma_spectrum(1.0, Q_SIGMA, ctx))
    u = 30.0 * _normal(200, SHAPE)
    sigma = core.sigma_field(u, ctx)
    assert not bool(jnp.any(jnp.isnan(sigma)))
    assert bool(jnp.all(sigma >= ctx.s_min))
    assert float(jnp.min(sigma - ctx.s_min)) < 1e-6, "fixture: u must drive exp() into underflow"


def test_21_sigma_field_at_zero_u_is_s_min_plus_one_exactly():
    ctx = core.make_ctx(_key(21), SHAPE)
    ctx = ctx._replace(sqrtS_sigma=core.sigma_spectrum(1.0, Q_SIGMA, ctx))
    sigma = core.sigma_field(jnp.zeros(SHAPE, dtype=jnp.float64), ctx)
    expected = jnp.full(SHAPE, ctx.s_min + 1.0, dtype=jnp.float64)
    assert np.array_equal(np.asarray(sigma), np.asarray(expected))


def test_22_sigma_field_shift_invariant_iff_dc_multiplier_zero():
    ctx = core.make_ctx(_key(22), SHAPE)
    s = core.sigma_spectrum(0.5, Q_SIGMA, ctx)
    s_dc = float(s[0, 0, 0])
    assert s_dc != 0.0, "fixture: the |q| <= q_sigma indicator contains q = 0"
    ctx_dc = ctx._replace(sqrtS_sigma=s)
    ctx_nodc = ctx._replace(sqrtS_sigma=s.at[0, 0, 0].set(0.0))
    u = _normal(220, SHAPE)
    shift = 3.0
    assert _rel(core.sigma_field(u + shift, ctx_nodc), core.sigma_field(u, ctx_nodc)) <= TOL
    shifted = core.sigma_field(u + shift, ctx_dc)
    base = core.sigma_field(u, ctx_dc)
    assert _rel(shifted, base) > 1e-3
    assert _rel(shifted - ctx.s_min, (base - ctx.s_min) * np.exp(shift * s_dc)) <= TOL


def test_23_sigma_spectrum_calibration():
    shape = (16, 16, 16)
    c = 0.7
    q_sigma = 3.0
    ctx = core.make_ctx(_key(23), shape)
    assert np.array_equal(
        np.asarray(core.sigma_spectrum(0.0, q_sigma, ctx)), np.zeros(_h(shape))
    ), "c == 0 must give sqrtS_sigma exactly 0"
    s = core.sigma_spectrum(c, q_sigma, ctx)
    assert s.shape == _h(shape) and s.dtype == jnp.float64
    inside = _index_radius(shape) <= q_sigma
    level = c * np.sqrt(np.prod(shape) / np.sum(_mult_ref(shape) * inside))
    s_np = np.asarray(s)
    assert np.all(s_np[~inside] == 0.0), "sqrtS_sigma must vanish outside |q| <= q_sigma"
    assert np.max(np.abs(s_np[inside] - level)) <= TOL * level, (
        "sqrtS_sigma must be flat at c*sqrt(N / sum(mult*indicator))"
    )
    fields = [core.apply_mult(_normal(2300 + i, shape), s) for i in range(64)]
    std = float(jnp.std(jnp.stack(fields)))
    assert abs(std - c) <= TOL_CALIBRATION * c, f"empirical std {std} vs c {c}"


def test_24_objective_finite_and_terms_nonnegative():
    state = _random_state(24)
    ctx = _full_ctx(24, state)
    J, terms = core.objective(state, ctx)
    assert bool(jnp.isfinite(J))
    assert float(terms.e_work) >= 0.0
    assert float(terms.e_pos) >= 0.0
    assert _rel(terms.e_z, 0.5 * jnp.sum(state.z ** 2)) <= TOL
    assert _rel(terms.e_u, 0.5 * jnp.sum(state.u ** 2)) <= TOL
    assert _rel(J, terms.e_work + ctx.mu * terms.e_pos + terms.e_z + terms.e_u) <= TOL


def _gradients(seed):
    state = _random_state(seed)
    ctx = _full_ctx(seed, state)
    auto = jax.grad(lambda s: core.objective(s, ctx)[0])(state)
    g_z, g_u = core.grad(state, ctx)
    return auto, g_z, g_u


def test_25_analytic_g_z_matches_autodiff():
    auto, g_z, _ = _gradients(25)
    assert _rel(g_z, auto.z) <= TOL


def test_26_analytic_g_u_matches_autodiff():
    auto, _, g_u = _gradients(26)
    assert _rel(g_u, auto.u) <= TOL


def test_27_autodiff_matches_central_finite_differences():
    shape = (8, 8, 8)
    state = _random_state(27, shape)
    ctx = _full_ctx(27, state)
    n = state.z.size

    @jax.jit
    def jflat(x):
        return core.objective(core.State(x[:n].reshape(shape), x[n:].reshape(shape)), ctx)[0]

    x = jnp.concatenate([state.z.ravel(), state.u.ravel()])
    g = jax.grad(jflat)(x)
    h = 1e-5
    for i in np.asarray(jax.random.choice(_key(270), 2 * n, (20,), replace=False)):
        e = jnp.zeros(2 * n, dtype=jnp.float64).at[int(i)].set(h)
        fd = float(jflat(x + e) - jflat(x - e)) / (2.0 * h)
        gi = float(g[int(i)])
        err = abs(fd - gi) / max(abs(fd), abs(gi))
        assert err < TOL_FD, f"coordinate {int(i)}: fd {fd} vs grad {gi}, rel {err:.2e}"


def test_28_no_nan_gradient_with_exact_zero_amplitudes():
    ctx = core.make_ctx(_key(28), SHAPE)
    F = core.project_band(core.r2c(_normal(280, SHAPE)), ctx.mask_obs)
    work_flat = np.flatnonzero(np.asarray(ctx.mask_work))
    zeros_flat = work_flat[::3]
    F = F.ravel().at[zeros_flat].set(0.0).reshape(_h(SHAPE))
    assert bool(jnp.all(F.ravel()[zeros_flat] == 0.0))
    k = core.fit_scale(F, ctx, ctx.mask_work)
    assert bool(jnp.isfinite(k)) and float(k) > 0.0, "fixture: F is non-zero elsewhere on mask_work"
    d = core.resid_coeff(F, ctx, k)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(d))))
    assert bool(jnp.all(d.ravel()[zeros_flat] == 0.0)), "resid_coeff must be 0 where |F| == 0"
    g_re, g_im = jax.grad(
        lambda re_, im_: core.e_work(re_ + 1j * im_, ctx, k), argnums=(0, 1)
    )(jnp.real(F), jnp.imag(F))
    assert bool(jnp.all(jnp.isfinite(g_re))) and bool(jnp.all(jnp.isfinite(g_im)))
    state = _random_state(281)
    ctx = _full_ctx(281, state)
    F_state = _F_of(state, ctx)
    assert bool(jnp.any(F_state[~ctx.mask_obs] == 0.0)), "fixture: sf() is exactly 0 off mask_obs"
    auto = jax.grad(lambda s: core.objective(s, ctx)[0])(state)
    g_z, g_u = core.grad(state, ctx)
    for name, g in (("auto z", auto.z), ("auto u", auto.u), ("g_z", g_z), ("g_u", g_u)):
        assert not bool(jnp.any(jnp.isnan(g))), name


def test_29_neg_part_zero_above_floor():
    ctx = core.make_ctx(_key(29), SHAPE)
    rho = ctx.rho_floor + jnp.abs(_normal(290, SHAPE))
    rho = rho.at[0, 0, 0].set(ctx.rho_floor)
    n = core.neg_part(rho, ctx)
    assert np.array_equal(np.asarray(n), np.zeros(SHAPE))


def test_30_neg_part_equals_deficit_below_floor():
    ctx = core.make_ctx(_key(30), SHAPE)
    rho = ctx.rho_floor + _normal(300, SHAPE)
    below = np.asarray(rho < ctx.rho_floor)
    assert below.any() and (~below).any()
    n = np.asarray(core.neg_part(rho, ctx))
    deficit = np.asarray(rho - ctx.rho_floor)
    assert np.max(np.abs(n[below] - deficit[below])) <= TOL * np.max(np.abs(deficit[below]))
    assert np.all(n[~below] == 0.0)


def test_31_objective_decreases_for_small_lr():
    state = _random_state(31)
    ctx = _full_ctx(31, state)
    J0 = float(core.objective(state, ctx)[0])
    lr0 = _lr(state, ctx, 310)
    for j in range(11):
        lr = lr0 * 2.0 ** -j
        J1 = float(core.objective(core.step(_copy(state), ctx, lr), ctx)[0])
        assert J1 < J0, f"lr = {lr:.3e} (0.5/L * 2^-{j}): J {J0} -> {J1}"


def test_32_null_g_z_vanishes_on_free_coefficients():
    ctx = _null_ctx(32)
    _assert_null_masks(ctx, SHAPE)
    _assert_constant_sigma(ctx, SHAPE)
    assert float(ctx.mu) == 0.0
    u = _normal(320, SHAPE)
    z_off_free = core.c2r(core.r2c(_normal(321, SHAPE)) * ~ctx.mask_free, SHAPE)
    for label, z in (("z = 0", jnp.zeros(SHAPE, dtype=jnp.float64)), ("z off mask_free", z_off_free)):
        state = core.State(z, u)
        g_z, _ = core.grad(state, ctx)
        auto = jax.grad(lambda s: core.objective(s, ctx)[0])(state)
        for name, g in (("analytic g_z", g_z), ("autodiff g_z", auto.z)):
            work = _band_max(g, ctx.mask_work)
            free = _band_max(g, ctx.mask_free)
            assert work > 0.0, f"{label}, {name}: gradient must be non-trivial on mask_work"
            assert free <= TOL_ROUNDTRIP * work, (
                f"{label}, {name}: free/work = {free / work:.3e}; coupling into mask_free. "
                f"{ROUNDTRIP_MSG}"
            )


def test_33_null_delta_rho_free_coefficients_zero_after_50_steps():
    ctx = _null_ctx(33)
    _assert_null_masks(ctx, SHAPE)
    _assert_constant_sigma(ctx, SHAPE)
    _, final, _ = _refine(ctx, 330)
    drho = core.delta_rho(final, ctx)
    work = _band_max(drho, ctx.mask_work)
    free = _band_max(drho, ctx.mask_free)
    assert work > NONZERO * float(jnp.max(jnp.abs(core.r2c(ctx.rho0)))), (
        "fixture: refinement must move the work coefficients"
    )
    assert free <= TOL_NULL * work, f"free/work = {free / work:.3e} after {N_NULL} steps"


def test_34_null_delta_r_free_zero():
    ctx = _null_ctx(34)
    _assert_null_masks(ctx, SHAPE)
    _assert_constant_sigma(ctx, SHAPE)
    state0, final, _ = _refine(ctx, 340)
    assert _r_fitted_on_work(ctx.F0, ctx, ctx.mask_free) > 1e-3, "fixture: R_free(F0) must be non-zero"
    d0 = float(core.delta_r_free(state0, ctx))
    d = float(core.delta_r_free(final, ctx))
    assert np.isfinite(d0) and np.isfinite(d)
    assert abs(d0) <= TOL_NULL, f"delta_r_free at the start state = {d0:.3e}"
    assert abs(d) <= TOL_NULL, f"delta_r_free after {N_NULL} steps = {d:.3e}"


def test_35_null_r_work_strictly_decreases():
    ctx = _null_ctx(35)
    _assert_null_masks(ctx, SHAPE)
    _assert_constant_sigma(ctx, SHAPE)
    state0, final, trace = _refine(ctx, 350)
    r0 = _r_fitted_on_work(_F_of(state0, ctx), ctx, ctx.mask_work)
    r1 = _r_fitted_on_work(_F_of(final, ctx), ctx, ctx.mask_work)
    assert r1 < r0, f"R_work {r0} -> {r1}"
    assert float(trace.r_work[-1]) < float(trace.r_work[0])


def test_36_nonconstant_sigma_couples_into_free_coefficients():
    ctx = _null_ctx(36, c=0.5)
    _assert_null_masks(ctx, SHAPE)
    assert float(jnp.max(ctx.sqrtS_sigma)) > 0.0
    _, final, _ = _refine(ctx, 360)
    drho = core.delta_rho(final, ctx)
    work = _band_max(drho, ctx.mask_work)
    free = _band_max(drho, ctx.mask_free)
    assert work > 0.0
    assert free > NONZERO * work, f"free/work = {free / work:.3e}; c > 0 must break the null"


def test_37_positivity_penalty_couples_into_free_coefficients():
    ctx = _null_ctx(37)
    _assert_null_masks(ctx, SHAPE)
    _assert_constant_sigma(ctx, SHAPE)
    state0 = _null_start(370)
    rho0 = core.rho_total(state0, ctx)
    ctx = ctx._replace(rho_floor=_f64(np.quantile(np.asarray(rho0), 0.25)))
    n0 = core.neg_part(rho0, ctx)
    g_data = core.grad_rho(state0, ctx)
    mu = float(jnp.linalg.norm(g_data.ravel()) / jnp.linalg.norm(n0.ravel()))
    assert np.isfinite(mu) and mu > 0.0, "fixture: mu balances e_pos against e_work"
    ctx = ctx._replace(mu=_f64(mu))
    assert float(core.e_pos(rho0, ctx)) > 0.0
    _, final, _ = _refine(ctx, 370)
    drho = core.delta_rho(final, ctx)
    work = _band_max(drho, ctx.mask_work)
    free = _band_max(drho, ctx.mask_free)
    assert work > 0.0
    assert free > NONZERO * work, f"free/work = {free / work:.3e}; mu > 0 must break the null"


def test_38_delta_r_free_vanishes_monotonically_as_c_decreases():
    cs = (0.1, 0.05, 0.025, 0.0125)
    base = _null_ctx(38)
    _assert_null_masks(base, SHAPE)
    _assert_constant_sigma(base, SHAPE)

    def with_c(c):
        return base._replace(sqrtS_sigma=core.sigma_spectrum(c, Q_SIGMA, base))

    lr = _lr(_null_start(380), with_c(cs[0]), 381)
    deltas = []
    for c in cs:
        ctx = with_c(c)
        _, final, _ = _refine(ctx, 380, lr=lr)
        deltas.append(abs(float(core.delta_r_free(final, ctx))))
    _, final0, _ = _refine(base, 380, lr=lr)
    d_null = abs(float(core.delta_r_free(final0, base)))
    assert all(np.isfinite(deltas))
    assert d_null <= TOL_NULL, f"c = 0 endpoint: |delta_r_free| = {d_null:.3e}"
    assert deltas[-1] > 1e3 * TOL_NULL, f"fixture: smallest c must be above the null floor, {deltas}"
    assert all(a > b for a, b in zip(deltas, deltas[1:])), f"not monotone in c: {deltas}"
    assert deltas[-1] <= 2.0 * (cs[-1] / cs[0]) * deltas[0], (
        f"slower than linear convergence to 0 within a factor 2: {deltas}"
    )


def test_39_identity_sym_projection_is_identity():
    H = _h(SHAPE)
    sym_idx, sym_phase = core.identity_sym(H)
    assert sym_idx.shape == (1,) + H and sym_idx.dtype == jnp.int32
    assert sym_phase.shape == (1,) + H and sym_phase.dtype == jnp.complex128
    F = _cnormal(39, H)
    assert np.array_equal(np.asarray(core.project_sym(F, sym_idx, sym_phase)), np.asarray(F))


def test_40_project_sym_output_invariant_under_each_operator():
    sym_idx, sym_phase = _twofold_sym(SHAPE)
    P = core.project_sym(_cnormal(40, _h(SHAPE)), sym_idx, sym_phase)
    for k in range(2):
        assert _rel(_apply_op(P, sym_idx, sym_phase, k), P) <= TOL, k


def test_41_delta_rho_spectrum_is_symmetric():
    state = _random_state(41)
    ctx = _full_ctx(41, state)
    assert ctx.sym_idx.shape[0] == 2
    D = core.r2c(core.delta_rho(state, ctx))
    assert _rel(core.project_sym(D, ctx.sym_idx, ctx.sym_phase), D) <= TOL


def test_42_fit_scale_recovers_k_exactly():
    ctx = core.make_ctx(_key(42), SHAPE)
    F = core.project_band(core.r2c(_normal(420, SHAPE)), ctx.mask_obs)
    k0 = 2.75
    mask = ctx.mask_work
    garbage = 1e3 * jnp.abs(_normal(421, _h(SHAPE)))
    ctx = ctx._replace(amp=jnp.where(mask, k0 * jnp.abs(F), garbage))
    k = core.fit_scale(F, ctx, mask)
    assert abs(float(k) - k0) <= TOL * k0
    dk = jax.grad(lambda re_: core.fit_scale(re_ + 1j * jnp.imag(F), ctx, mask))(jnp.real(F))
    assert np.array_equal(np.asarray(dk), np.zeros(_h(SHAPE))), "fit_scale must stop_gradient k"


def test_43_r_factor_zero_for_exact_fit():
    ctx = core.make_ctx(_key(43), SHAPE)
    F = core.project_band(core.r2c(_normal(430, SHAPE)), ctx.mask_obs)
    k = 1.7
    garbage = 1e3 * jnp.abs(_normal(431, _h(SHAPE)))
    for mask in (ctx.mask_work, ctx.mask_free):
        c = ctx._replace(amp=jnp.where(mask, k * jnp.abs(F), garbage))
        assert abs(float(core.r_factor(F, c, k, mask))) <= TOL


def test_44_r_factor_invariant_to_joint_scaling_of_amp_and_k():
    ctx = core.make_ctx(_key(44), SHAPE)
    F = core.project_band(core.r2c(_normal(440, SHAPE)), ctx.mask_obs)
    k = float(core.fit_scale(F, ctx, ctx.mask_work))
    lam = 3.7
    scaled = ctx._replace(amp=lam * ctx.amp)
    for mask in (ctx.mask_work, ctx.mask_free):
        r1 = float(core.r_factor(F, ctx, k, mask))
        r2 = float(core.r_factor(F, scaled, lam * k, mask))
        assert r1 > 0.0
        assert abs(r1 - r2) <= TOL * r1


def test_45_eta_c_finite_and_zero_in_null():
    ctx = _null_ctx(45)
    _assert_null_masks(ctx, SHAPE)
    _assert_constant_sigma(ctx, SHAPE)
    state0, final, _ = _refine(ctx, 450)
    dr_work = _r_fitted_on_work(_F_of(state0, ctx), ctx, ctx.mask_work) - _r_fitted_on_work(
        _F_of(final, ctx), ctx, ctx.mask_work
    )
    assert dr_work > 0.0, "fixture: R_work must decrease so eta_c is defined"
    eta = float(core.eta_c(final, ctx))
    assert np.isfinite(eta)
    assert abs(eta) <= TOL_NULL / dr_work, (
        f"eta_c = {eta:.3e}; bound is the test-34 null tolerance over the R_work decrease {dr_work:.3e}"
    )


CUDA_TABLE = ((580, 13), (525, 12))


@gpu
def test_46_gpu_backend_under_cuda_platform():
    import device

    if os.environ.get("JAX_PLATFORMS") != "cuda":
        pytest.skip("JAX_PLATFORMS=cuda not set")
    assert jax.default_backend() == "gpu"
    device.require_accelerator("gpu")
    with pytest.raises(Exception):
        device.require_accelerator("cpu")


@gpu
def test_47_cuda_major_total():
    import device

    assert device.cuda_major("580.65.06", CUDA_TABLE) == 13
    assert device.cuda_major("595.10", CUDA_TABLE) == 13
    assert device.cuda_major("535.104.05", CUDA_TABLE) == 12
    assert device.cuda_major("535.104.05", ((500, 11),)) == 11, "table must be a parameter"
    for bad in ("470.82.01", "not-a-driver", "5x5.1", ""):
        with pytest.raises(ValueError, match=re.escape(bad)):
            device.cuda_major(bad, CUDA_TABLE)
    assert device.cuda_major(device.probe_driver(), CUDA_TABLE) in (12, 13)


@gpu
def test_48_gpu_and_cpu_objective_agree():
    state = _random_state(48)
    ctx = _full_ctx(48, state)
    J, terms = core.objective(state, ctx)
    gpu_values = [float(J)] + [float(t) for t in terms]
    probe = (
        "import jax, core, test_core as t\n"
        "assert jax.default_backend() == 'cpu'\n"
        "s = t._random_state(48)\n"
        "c = t._full_ctx(48, s)\n"
        "J, terms = core.objective(s, c)\n"
        "print(' '.join(float(v).hex() for v in (J,) + tuple(terms)))\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    cpu_values = [float.fromhex(v) for v in out.stdout.split()]
    names = ("J",) + terms._fields
    assert len(cpu_values) == len(gpu_values)
    for name, g, c in zip(names, gpu_values, cpu_values):
        assert abs(g - c) <= TOL * max(abs(g), abs(c)), f"{name}: gpu {g} vs cpu {c}"


@gpu
def test_49_run_repeatable():
    state = _random_state(49)
    ctx = _full_ctx(49, state)
    lr = _lr(state, ctx, 490)
    a = core.run(_copy(state), ctx, lr, 10)
    b = core.run(_copy(state), ctx, lr, 10)
    bitwise = jax.default_backend() == "cpu" or (
        "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "")
    )
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert _rel(x, y) <= TOL_NULL
        if bitwise:
            assert np.array_equal(np.asarray(x), np.asarray(y)), "deterministic run must be bitwise"
