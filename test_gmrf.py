from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import core

from fieldrefine.precision import estimate, gmrf

# Interfaces pinned here that GMRF_SPEC S6-S7 does not name, one per line, with the
# test that forces each. The author may rename any of them before implementation.
#   estimate.GCtx                 S4 input contract; core.Ctx fields + inv_S, phi,
#                                 rf_seed, mu_graph, m, kernel, chunk, knn. core.Ctx.mu
#                                 keeps its meaning, the mu_pos of S5; mu_graph is the
#                                 coupling weight S4 calls mu. m/kernel/chunk/rf_seed
#                                 must be static, not traced.
#   estimate.P_graph(x, phi, seed, m, mu, kernel, chunk)   S6 lists the first five
#   estimate.P_graph_knn(x, idx, w, mu)          the sparse path of S7, test 5
#   estimate.solve(x0, ctx, eps_floor, max_iter) -> (x, GTrace)   tests 5, 9-19, 22-23
#   estimate.GTrace(j, e_work, e_pos, x_p_x, r_work, r_free, gnorm, t_star, at_floor)
#   estimate.delta_r_free(x, ctx)                the readout on x, tests 5, 9-13, 16, 19
#   estimate.memory_report(n, d, m, chunk, budget)  test 21; raises ValueError over budget
#   estimate.hypergrad_unrolled(params, ctx, eps_floor, max_iter, window)  tests 22, 23
#   estimate.hypergrad_implicit(params, ctx, eps_floor, max_iter)          test 23
#   gmrf.graph_spectrum(apply_L, n_eig, key, shape)  S7 omits the start-vector shape
# solve must be differentiable w.r.t. ctx.phi and must not donate its arguments.
# solve runs to t* = min{t: R_work <= eps_floor} or to max_iter, whichever comes first,
# and returns x at t*; eps_floor = 0.0 therefore means a fixed run of max_iter steps.
# Both hypergrad entries recompute phi = embed_cnn(params, ctx.rho0) and ignore ctx.phi.
# Row i of phi is voxel x.ravel()[i], C order.

SHAPE = (8, 10, 12)
DENSE_SHAPE = (4, 4, 4)
PATCH = 3
M = 64
M_DENSE = 32
M_SPARSE = 256
K_NN = 256
CHUNK = 97
CHUNK_DENSE = 13
KERNEL = "gaussian"
STATIONARY = ("gaussian", "rq")
S_FLOOR = 1e-2
SD_POS = 4.0
FLOOR_FRAC = 0.5
MAX_ITER = 50
MAX_ITER_LONG = 200
N_EIG = 24
D_OUT = 4
N_UNROLL = 12
N_CONVERGED = 150
FD_H = 1e-6

TOL = 1e-10
TOL_CHUNK = 1e-12
TOL_NULL = 1e-12
TOL_POSITION = 1e-8
TOL_IMPLICIT = 1e-6
# the line search safeguards (max on curvature, descent restart) are switches, not
# smooth functions; they flip between the two perturbed solves and bound the agreement
TOL_FD = 1e-2
# squaring the kernel to force w >= 0 costs a square root in the harmonic count, so the
# low-rank kernel carries O(1/sqrt(sqrt(m))) error against the exact one the kNN path uses
TOL_SPARSE = 0.75
TOL_MEMORY = 0.20
NONZERO = 1e-6
KAPPA_ZERO = 1e-8
ABOVE_FLOOR = 1e3 * TOL_NULL

SELF_ADJOINT_MSG = (
    "P_apply = Pi(P_stat + P_graph) is self-adjoint on the range of Pi, which is where "
    "the variable lives (S5: Pi x = x); it is not self-adjoint on unprojected fields, "
    "since Pi does not commute with P_stat + P_graph. Both arguments are Pi-projected."
)
STATIONARY_MSG = (
    "position-only phi must give a graph that depends on the voxel separation alone, "
    "and on a unit cell that separation is toroidal. Three ways this breaks: random "
    "frequencies drawn from a continuous measure make Z Z^T an O(m^-1/2) approximation "
    "to a stationary kernel rather than an exactly stationary operator; a feature "
    "distance that is not periodic makes W Toeplitz, not circulant, so |0.99 - 0.01| "
    "reads 0.98 instead of 0.02; a degree summed over a non-toroidal neighbourhood "
    "varies at the cell boundary. A larger m fixes none of them."
)
CONSTANT_GRAD_MSG = (
    "P_graph(1) must be exactly 0, not small: S6 prescribes that d is computed by the "
    "same two-pass structure as the matvec with x = ones, so d*1 and Z(Z^T 1) are the "
    "same floating-point expression. A non-zero value means degree and the matvec sum "
    "in different orders, or over different chunks."
)
FREEZE_MSG = (
    "S7 freeze rule: phi, rf_seed and d are computed once from rho0 and are bitwise "
    "constant for the whole solve. Recomputing them from a density already fitted to "
    "the work reflections makes the prior data-dependent and destroys the null."
)
UNROLL_MSG = (
    "the finite-difference check pins window == max_iter, the untruncated case. With a "
    "shorter window the unrolled hypergradient differs from the exact derivative by the "
    "truncation, which is intended, so the two are not comparable there."
)
NCS_MSG = (
    "the NCS fixture is translational: raw patch features are not rotation-equivariant, "
    "so a patch and its rotated mate are different vectors and a rotational NCS is not "
    "detectable by this embedding. R = I, t an integer voxel shift."
)


def _key(seed):
    return jax.random.PRNGKey(seed)


def _n(shape):
    return int(np.prod(shape))


def _f64(v):
    return jnp.asarray(v, dtype=jnp.float64)


def _normal(seed, shape):
    return jax.random.normal(_key(seed), shape, dtype=jnp.float64)


def _rel(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    scale = float(jnp.maximum(jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))))
    diff = float(jnp.max(jnp.abs(a - b)))
    return diff if scale == 0.0 else diff / scale


def _ip_real(x, y):
    return float(jnp.sum(x * y))


def _ip_mult(F, G, mult):
    return float(jnp.sum(mult * jnp.real(F * jnp.conj(G))))


def _core_view(g):
    return core.Ctx(**{f: getattr(g, f) for f in core.Ctx._fields})


def _gctx(seed, shape=SHAPE, m=M, chunk=CHUNK, kernel=KERNEL, phi=None, **overrides):
    ctx = core.make_ctx(_key(seed), shape)
    inv_S = 1.0 / (jnp.asarray(ctx.sqrtS) ** 2 + S_FLOOR)
    if phi is None:
        phi = gmrf.concat_features(gmrf.patch_features(ctx.rho0, PATCH, 1), whiten=True)
    g = estimate.GCtx(
        **ctx._asdict(), inv_S=inv_S, phi=phi, rf_seed=1000 + seed,
        mu_graph=_f64(0.0), m=m, kernel=kernel, chunk=chunk, knn=None,
    )
    return g._replace(**overrides)


def _pi(x, g):
    return core.pi_real(x, _core_view(g))


def _graph(x, g, mu, kernel=None, chunk=None):
    return estimate.P_graph(
        x, g.phi, g.rf_seed, g.m, mu,
        g.kernel if kernel is None else kernel,
        g.chunk if chunk is None else chunk,
    )


def _mu_balanced(g, seed):
    x = _pi(_normal(seed, g.rho0.shape), g)
    q_stat = _ip_real(x, estimate.P_stat(x, g.inv_S))
    q_graph = _ip_real(x, _graph(x, g, 1.0))
    assert q_stat > 0.0, f"fixture: x'P_stat x = {q_stat:.3e} must be positive"
    assert q_graph > 0.0, f"fixture: x'Lx = {q_graph:.3e} must be positive and non-trivial"
    return q_stat / q_graph


def _zeros(shape=SHAPE):
    return jnp.zeros(shape, dtype=jnp.float64)


def _r_fitted(F, g, mask):
    cv = _core_view(g)
    k = core.fit_scale(F, cv, g.mask_work)
    return float(core.r_factor(F, cv, k, mask))


def _r_work0(g):
    return _r_fitted(core.sf(g.rho0, _core_view(g)), g, g.mask_work)


def _eps_floor(g):
    return FLOOR_FRAC * _r_work0(g)


def _reachable_floor(g, max_iter=MAX_ITER_LONG):
    # the prior bounds how far R_work can fall; a floor below that minimum is unreachable
    _, pilot = estimate.solve(_zeros(g.rho0.shape), g, 0.0, max_iter)
    lo = float(pilot.r_work[max_iter])
    hi = float(pilot.r_work[0])
    assert lo < hi, f"fixture: the solve must reduce R_work, {hi} -> {lo}"
    return lo + 0.25 * (hi - lo)


def _solve(g, eps_floor=None, max_iter=MAX_ITER):
    eps = _eps_floor(g) if eps_floor is None else eps_floor
    return estimate.solve(_zeros(g.rho0.shape), g, eps, max_iter)


def _drf(g, eps_floor=None, max_iter=MAX_ITER):
    x, trace = _solve(g, eps_floor, max_iter)
    return float(estimate.delta_r_free(x, g)), x, trace


def _band_max(x, mask):
    return float(jnp.max(jnp.where(mask, jnp.abs(core.r2c(x)), 0.0)))


def _conj_planes(shape):
    return [0] + ([shape[2] // 2] if shape[2] % 2 == 0 else [])


def _reflect(a, shape):
    a = np.asarray(a)
    out = a.copy()
    i = (-np.arange(shape[0])) % shape[0]
    j = (-np.arange(shape[1])) % shape[1]
    for p in _conj_planes(shape):
        out[:, :, p] = a[i][:, j, p]
    return out


def _assert_null_fixture(g, shape=SHAPE):
    work = np.asarray(g.mask_work)
    free = np.asarray(g.mask_free)
    assert work.any() and free.any(), "fixture: mask_work and mask_free must be non-empty"
    assert not (work & free).any(), "fixture: mask_work and mask_free must be disjoint"
    assert np.array_equal(np.asarray(g.mask_obs), work | free), "fixture: mask_obs != work | free"
    assert np.array_equal(work, _reflect(work, shape)), "fixture: mask_work not Friedel-symmetric"
    assert np.array_equal(free, _reflect(free, shape)), "fixture: mask_free not Friedel-symmetric"
    assert g.sym_idx.shape[0] == 1, "fixture: the null configuration uses identity_sym"
    assert float(g.mu) == 0.0, "fixture: mu_pos must be 0; E_pos couples work to free"
    assert bool(jnp.all(g.inv_S > 0.0)), "S4: inv_S must be strictly positive everywhere"
    assert _r_fitted(g.F0, g, g.mask_free) > 1e-3, "fixture: R_free(F0) must be non-zero"


def _assert_moved(x, g, label):
    work = _band_max(x, g.mask_work)
    assert work > 0.0, f"{label}: the solve must move the work coefficients"
    return work


def _dense_w(phi, seed, m, kernel):
    Z = np.asarray(gmrf.rf_block(phi, seed, m, kernel), dtype=np.float64)
    return Z @ Z.T


def _cnn_params(seed):
    k1, k2 = jax.random.split(_key(seed))
    return {
        "w": 0.1 * jax.random.normal(k1, (PATCH, PATCH, PATCH, D_OUT), dtype=jnp.float64),
        "b": 0.1 * jax.random.normal(k2, (D_OUT,), dtype=jnp.float64),
    }


def _phase_b_ctx(g, params):
    return g._replace(phi=gmrf.embed_cnn(params, g.rho0))


def _phase_b_loss(params, g, eps_floor, max_iter):
    gg = _phase_b_ctx(g, params)
    x, _ = estimate.solve(_zeros(g.rho0.shape), gg, eps_floor, max_iter)
    return estimate.delta_r_free(x, gg)


def test_01_graph_quadratic_form_equals_the_explicit_pair_sum():
    g = _gctx(1, DENSE_SHAPE, m=M_DENSE, chunk=CHUNK_DENSE)
    n = _n(DENSE_SHAPE)
    assert g.phi.shape[0] == n, f"phi must carry one row per voxel in C order, got {g.phi.shape}"
    W = _dense_w(g.phi, g.rf_seed, g.m, g.kernel)
    assert _rel(W, W.T) <= TOL_CHUNK, "W = Z Z^T must be symmetric"
    d = np.asarray(gmrf.degree(g.phi, g.rf_seed, g.m, g.kernel, g.chunk))
    assert _rel(d, W.sum(axis=1)) <= TOL_CHUNK, (
        "degree must be the full row sum of W, self term included; the pair-sum identity "
        "holds only for that degree"
    )
    x = _normal(101, DENSE_SHAPE)
    xf = np.asarray(x).ravel()
    pair = 0.5 * float(np.sum(W * (xf[:, None] - xf[None, :]) ** 2))
    quad = _ip_real(x, _graph(x, g, 1.0))
    assert pair > 0.0, "fixture: the pair sum must be non-trivial"
    assert abs(quad - pair) <= TOL * pair, (
        f"x'Lx = {quad!r} but 0.5*sum_ij w_ij (x_i - x_j)^2 = {pair!r}, "
        f"relative {abs(quad - pair) / pair:.3e}"
    )


def test_02_p_graph_is_the_gradient_of_the_graph_energy():
    g = _gctx(2, DENSE_SHAPE, m=M_DENSE, chunk=CHUNK_DENSE)
    W = jnp.asarray(_dense_w(g.phi, g.rf_seed, g.m, g.kernel))
    mu = 0.75

    def energy(x):
        xf = x.ravel()
        # 0.5 x'(mu L)x written over ordered pairs
        return 0.25 * mu * jnp.sum(W * (xf[:, None] - xf[None, :]) ** 2)

    x = _normal(201, DENSE_SHAPE)
    auto = jax.grad(energy)(x)
    ana = _graph(x, g, mu)
    assert float(jnp.max(jnp.abs(auto))) > 0.0, "fixture: the energy gradient must be non-trivial"
    assert _rel(ana, auto) <= TOL, (
        f"P_graph differs from jax.grad of the graph energy by {_rel(ana, auto):.3e} relative"
    )


def test_03_constant_field_gives_zero_graph_energy_and_gradient():
    g = _gctx(3, DENSE_SHAPE, m=M_DENSE, chunk=CHUNK_DENSE)
    x = jnp.full(DENSE_SHAPE, 2.0, dtype=jnp.float64)
    out = _graph(x, g, 0.75)
    assert np.array_equal(np.asarray(out), np.zeros(DENSE_SHAPE)), (
        f"max |P_graph(const)| = {float(jnp.max(jnp.abs(out))):.3e}. {CONSTANT_GRAD_MSG}"
    )
    assert _ip_real(x, out) == 0.0, "the graph energy of a constant field must be exactly 0"
    probe = _normal(301, DENSE_SHAPE)
    assert float(jnp.max(jnp.abs(_graph(probe, g, 0.75)))) > 0.0, (
        "fixture: the graph operator must be non-trivial on a non-constant field"
    )


def test_04_chunked_matvec_equals_the_unchunked_result():
    g = _gctx(4)
    n = _n(SHAPE)
    Z = gmrf.rf_block(g.phi, g.rf_seed, g.m, g.kernel)
    assert Z.shape == (n, g.m), f"rf_block must return (rows, m), got {Z.shape}"
    x = _normal(401, SHAPE)
    xf = x.ravel()
    s_ref = Z.T @ xf
    y_ref = Z @ s_ref
    d_ref = Z @ (Z.T @ jnp.ones(n, dtype=jnp.float64))
    assert float(jnp.max(jnp.abs(y_ref))) > 0.0, "fixture: Z Z^T x must be non-trivial"
    for chunk in (7, 97, n):
        d = gmrf.degree(g.phi, g.rf_seed, g.m, g.kernel, chunk)
        y = d * xf - _graph(x, g, 1.0, chunk=chunk).ravel()
        assert _rel(d, d_ref) <= TOL_CHUNK, (
            f"chunk {chunk}: degree differs from the unchunked two-pass result by "
            f"{_rel(d, d_ref):.3e} relative"
        )
        assert _rel(y, y_ref) <= TOL_CHUNK, (
            f"chunk {chunk} ({n} rows, remainder {n % chunk}): Z(Z^T x) differs from the "
            f"unchunked result by {_rel(y, y_ref):.3e} relative"
        )


def test_05_low_rank_and_sparse_paths_agree_on_delta_r_free():
    g = _gctx(5, m=M_SPARSE)
    _assert_null_fixture(g)
    idx, w = gmrf.knn_graph(g.phi, K_NN)
    assert idx.shape == (_n(SHAPE), K_NN), f"knn_graph idx must be (N, k), got {idx.shape}"
    assert w.shape == idx.shape, f"knn_graph w must match idx, got {w.shape}"
    a = _pi(_normal(501, SHAPE), g)
    b = _pi(_normal(502, SHAPE), g)
    q_lr = _ip_real(a, _graph(a, g, 1.0))
    q_sp = _ip_real(a, estimate.P_graph_knn(a, idx, w, 1.0))
    assert q_lr > 0.0 and q_sp > 0.0, f"both paths must be PSD and non-trivial: {q_lr}, {q_sp}"
    cross = _ip_real(estimate.P_graph_knn(a, idx, w, 1.0), b)
    assert abs(cross - _ip_real(a, estimate.P_graph_knn(b, idx, w, 1.0))) <= TOL * abs(cross), (
        "the sparse path must be self-adjoint; W must be symmetrised before D - W is formed, "
        "since a k-nearest-neighbour relation is not symmetric"
    )
    mu = _mu_balanced(g, 503)
    # the two paths are matched in coupling strength, not in mu
    g_lr = g._replace(mu_graph=_f64(mu))
    g_sp = g._replace(mu_graph=_f64(mu * q_lr / q_sp), knn=(idx, w))
    eps = _eps_floor(g)
    d_lr, x_lr, _ = _drf(g_lr, eps)
    d_sp, x_sp, _ = _drf(g_sp, eps)
    _assert_moved(x_lr, g, "low-rank")
    _assert_moved(x_sp, g, "sparse")
    assert abs(d_lr) > ABOVE_FLOOR, (
        f"fixture: the low-rank path must give a reading above the null floor, got {d_lr:.3e}"
    )
    span = max(abs(d_lr), abs(d_sp))
    assert abs(d_lr - d_sp) <= TOL_SPARSE * span, (
        f"delta_r_free low-rank {d_lr:.6e} vs sparse {d_sp:.6e}, "
        f"{abs(d_lr - d_sp) / span:.3%} apart at matched coupling strength"
    )


def test_06_p_apply_self_adjoint_under_the_multiplicity_inner_product():
    g = _gctx(6)
    g = g._replace(mu_graph=_f64(_mu_balanced(g, 601)))
    x = _pi(_normal(602, SHAPE), g)
    y = _pi(_normal(603, SHAPE), g)
    px = estimate.P_apply(x, g)
    py = estimate.P_apply(y, g)
    a = _ip_mult(core.r2c(px), core.r2c(y), g.mult)
    b = _ip_mult(core.r2c(x), core.r2c(py), g.mult)
    span = max(abs(a), abs(b))
    assert span > 0.0, "fixture: the bilinear form must be non-trivial"
    assert abs(a - b) <= TOL * span, (
        f"<Px, y>_mult = {a!r} but <x, Py>_mult = {b!r}, relative {abs(a - b) / span:.3e}. "
        f"{SELF_ADJOINT_MSG}"
    )
    c = _ip_real(px, y)
    d = _ip_real(x, py)
    assert abs(c - d) <= TOL * max(abs(c), abs(d)), (
        f"real-space form disagrees: {c!r} vs {d!r}; by Parseval it must match the "
        f"multiplicity form"
    )
    assert abs(a - c) <= TOL * span, (
        f"Parseval broken between the two inner products: {a!r} vs {c!r}; the multiplicity "
        f"factor is missing or doubled"
    )


def test_07_p_apply_positive_definite():
    g = _gctx(7)
    g = g._replace(mu_graph=_f64(_mu_balanced(g, 701)))
    lo = float(jnp.min(jnp.where(g.mask_obs, g.inv_S, jnp.inf)))
    assert lo > 0.0, "S4: inv_S must be strictly positive; a zero makes P singular"
    worst = None
    for i in range(100):
        x = _pi(_normal(7000 + i, SHAPE), g)
        nsq = _ip_real(x, x)
        assert nsq > 0.0, f"draw {i}: the projected field must be non-zero"
        q = _ip_real(x, estimate.P_apply(x, g))
        q_graph = _ip_real(x, _graph(x, g, g.mu_graph))
        assert q > 0.0, f"draw {i}: x'Px = {q:.6e} is not positive"
        assert q_graph >= -TOL * abs(q), (
            f"draw {i}: x'(mu L)x = {q_graph:.6e} is negative; L = D - W is PSD by "
            f"x'Lx = 0.5 sum w_ij (x_i - x_j)^2, so a negative value is a sign error in "
            f"the degree term, which gives an indefinite P that diverges rather than "
            f"failing cleanly"
        )
        ratio = q / (lo * nsq)
        worst = ratio if worst is None else min(worst, ratio)
    assert worst >= 1.0 - TOL, (
        f"x'Px >= min(inv_S) ||x||^2 must hold on the band (Parseval plus a PSD graph "
        f"term); worst ratio over 100 draws {worst:.6f}"
    )


def test_08_mu_zero_reduces_p_apply_to_p_stat():
    g = _gctx(8)
    x = _pi(_normal(801, SHAPE), g)
    ref = _pi(estimate.P_stat(x, g.inv_S), g)
    got = estimate.P_apply(x, g._replace(mu_graph=_f64(0.0)))
    assert np.array_equal(np.asarray(got), np.asarray(ref)), (
        f"mu = 0 must reduce P_apply to Pi P_stat exactly, not to within "
        f"{_rel(got, ref):.3e}; the graph branch must contribute an exact zero"
    )
    assert _rel(estimate.P_stat(x, g.inv_S), core.apply_mult(x, g.inv_S)) <= TOL_CHUNK, (
        "P_stat must be c2r(r2c(x) * inv_S)"
    )
    on = g._replace(mu_graph=_f64(_mu_balanced(g, 802)))
    assert _rel(estimate.P_apply(x, on), ref) > NONZERO, (
        "fixture: mu > 0 must change P_apply, or test 8 asserts nothing"
    )


def test_09_null_mu_zero_gives_zero_delta_r_free():
    g = _gctx(9)
    _assert_null_fixture(g)
    eps = _eps_floor(g)
    off = g._replace(mu_graph=_f64(0.0))
    d, x, trace = _drf(off, eps)
    work = _assert_moved(x, off, "mu = 0")
    free = _band_max(x, off.mask_free)
    assert float(trace.r_work[int(trace.t_star)]) < _r_work0(off), (
        "fixture: the solve must reduce R_work, or there is nothing for the null to be "
        "measured against"
    )
    assert free <= TOL_NULL * work, (
        f"free/work = {free / work:.3e} in the refined x. With mu = 0 the precision is "
        f"P_stat alone, a Fourier multiplier, so a free coefficient starting at 0 receives "
        f"no gradient and must stay exactly 0. A small non-zero value is a coupling to "
        f"locate: the scale fitted across mask_obs rather than mask_work, a real-space "
        f"operation applied outside Pi, or mu_pos > 0"
    )
    assert abs(d) <= TOL_NULL, f"delta_r_free = {d:.6e} with mu = 0"
    on = g._replace(mu_graph=_f64(_mu_balanced(g, 901)))
    d_on, _, _ = _drf(on, eps)
    assert abs(d_on) > ABOVE_FLOOR, (
        f"fixture: the same configuration with mu > 0 must give a reading above the null "
        f"floor, got {d_on:.3e}; otherwise test 9 would pass with a broken readout"
    )


def test_10_null_constant_phi_gives_zero_delta_r_free():
    n = _n(SHAPE)
    g = _gctx(10, phi=jnp.ones((n, 8), dtype=jnp.float64))
    _assert_null_fixture(g)
    Z = gmrf.rf_block(g.phi, g.rf_seed, g.m, g.kernel)
    assert _rel(Z, jnp.broadcast_to(Z[:1], Z.shape)) == 0.0, (
        "constant phi must give bitwise identical feature rows, hence W = c 11^T"
    )
    g = g._replace(mu_graph=_f64(_mu_balanced(g, 1001)))
    eps = _eps_floor(g)
    d, x, _ = _drf(g, eps)
    work = _assert_moved(x, g, "constant phi")
    free = _band_max(x, g.mask_free)
    assert free <= TOL_NULL * work, (
        f"free/work = {free / work:.3e}. Constant phi gives W = c 11^T and degree c N, so "
        f"L = c(N I - 11^T): it annihilates the mean and is c N elsewhere, a Fourier "
        f"multiplier. It cannot move a coefficient between masks"
    )
    assert abs(d) <= TOL_NULL, f"delta_r_free = {d:.6e} with constant phi"
    varied = _gctx(10)
    varied = varied._replace(mu_graph=_f64(_mu_balanced(varied, 1002)))
    d_on, _, _ = _drf(varied, _eps_floor(varied))
    assert abs(d_on) > ABOVE_FLOOR, (
        f"fixture: a phi that reads rho0 must give a reading above the null floor at the "
        f"same coupling strength, got {d_on:.3e}"
    )


def test_11_null_position_features_give_zero_delta_r_free():
    n = _n(SHAPE)
    pos = SD_POS * gmrf.position_features(SHAPE)
    assert pos.shape[0] == n, f"position_features must give one row per voxel, got {pos.shape}"
    shift = (3, 4, 5)
    probe = _normal(1101, SHAPE)
    reported = {}
    for kernel in STATIONARY:
        g = _gctx(11, kernel=kernel, phi=pos)
        _assert_null_fixture(g)
        d_deg = np.asarray(gmrf.degree(g.phi, g.rf_seed, g.m, g.kernel, g.chunk))
        spread = float(np.max(d_deg) - np.min(d_deg)) / float(np.mean(d_deg))
        assert spread <= TOL_CHUNK, (
            f"{kernel}: the degree varies by {spread:.3e} across the grid. On a torus every "
            f"voxel has the same neighbourhood, so a position-only graph has a constant "
            f"degree. {STATIONARY_MSG}"
        )
        lx = _graph(probe, g, 1.0)
        rolled = _graph(jnp.roll(probe, shift, axis=(0, 1, 2)), g, 1.0)
        assert float(jnp.max(jnp.abs(lx))) > 0.0, f"{kernel}: fixture: L must be non-trivial"
        assert _rel(rolled, jnp.roll(lx, shift, axis=(0, 1, 2))) <= TOL, (
            f"{kernel}: L does not commute with the lattice translation {shift}, by "
            f"{_rel(rolled, jnp.roll(lx, shift, axis=(0, 1, 2))):.3e} relative. A graph "
            f"that commutes with every translation is a convolution, hence a multiplier. "
            f"{STATIONARY_MSG}"
        )
        g = g._replace(mu_graph=_f64(_mu_balanced(g, 1102)))
        d, x, _ = _drf(g)
        work = _assert_moved(x, g, kernel)
        free = _band_max(x, g.mask_free)
        reported[kernel] = d
        assert free <= TOL_POSITION * work, (
            f"{kernel}: free/work = {free / work:.3e} in the refined x"
        )
        assert abs(d) <= TOL_POSITION, (
            f"{kernel}: delta_r_free = {d:.6e}, above 1e-8 in float64. This is the "
            f"empirical form of the admissibility theorem of S7: either the theorem is "
            f"wrong or something couples unintendedly. {STATIONARY_MSG}"
        )
    inner = _gctx(11, kernel="inner", phi=pos)
    inner = inner._replace(mu_graph=_f64(_mu_balanced(inner, 1103)))
    d_inner, _, _ = _drf(inner)
    assert abs(d_inner) > NONZERO, (
        f"fixture: the inner-product kernel on the same positions gives w_ij = <r_i, r_j>, "
        f"which is not a function of r_i - r_j, so it is outside the theorem and must give "
        f"a non-zero reading; got {d_inner:.3e}. Without this the null tests above would "
        f"pass with a readout that is always zero"
    )
    patch = _gctx(11)
    patch = patch._replace(mu_graph=_f64(_mu_balanced(patch, 1104)))
    d_patch, _, _ = _drf(patch)
    assert abs(d_patch) > ABOVE_FLOOR, (
        f"fixture: a phi that reads rho0 must give a non-zero reading, got {d_patch:.3e}"
    )
    print(
        f"test 11 position-only delta_r_free: "
        + ", ".join(f"{k} {v:.3e}" for k, v in reported.items())
        + f"; controls inner {d_inner:.3e}, patch {d_patch:.3e}"
    )


def test_12_delta_r_free_vanishes_monotonically_as_sd_phi_decreases():
    sds = (0.4, 0.2, 0.1, 0.05)
    base = _gctx(12)
    _assert_null_fixture(base)
    unit = base.phi / float(jnp.std(base.phi))
    # fixed iteration count: t* moves with the prior and test 19 covers that separately
    eps = 0.0

    def with_sd(sd):
        g = base._replace(phi=unit * sd)
        return g._replace(mu_graph=_f64(_mu_balanced(base, 1201)))

    deltas = [abs(_drf(with_sd(sd), eps)[0]) for sd in sds]
    zero = base._replace(
        phi=jnp.zeros_like(unit), mu_graph=_f64(_mu_balanced(base, 1201))
    )
    d_zero = abs(_drf(zero, eps)[0])
    assert all(np.isfinite(deltas)), f"delta_r_free not finite over sd(phi): {deltas}"
    assert d_zero <= TOL_NULL, (
        f"sd(phi) = 0 endpoint: |delta_r_free| = {d_zero:.3e}; a constant phi is the null "
        f"of test 10 and must read exactly zero"
    )
    assert deltas[-1] > ABOVE_FLOOR, (
        f"fixture: the smallest sd(phi) must stay above the null floor, {deltas}"
    )
    assert all(a > b for a, b in zip(deltas, deltas[1:])), (
        f"not monotone in sd(phi): {deltas} for {sds}"
    )
    assert deltas[-1] <= 2.0 * (sds[-1] / sds[0]) * deltas[0], (
        f"slower than linear approach to the null within a factor 2: {deltas} for {sds}; "
        f"w_ij departs from constant at O(sd^2), so the decay should be faster than linear"
    )


def test_13_delta_r_free_vanishes_monotonically_as_mu_decreases():
    base = _gctx(13)
    _assert_null_fixture(base)
    unit = _mu_balanced(base, 1301)
    mus = tuple(unit * f for f in (1.0, 0.1, 0.01, 0.001))
    # fixed iteration count: t* moves with mu and test 19 covers that separately
    eps = 0.0
    deltas = [abs(_drf(base._replace(mu_graph=_f64(mu)), eps)[0]) for mu in mus]
    d_zero = abs(_drf(base._replace(mu_graph=_f64(0.0)), eps)[0])
    assert all(np.isfinite(deltas)), f"delta_r_free not finite over mu: {deltas}"
    assert d_zero <= TOL_NULL, (
        f"mu = 0 endpoint: |delta_r_free| = {d_zero:.3e}; this is the null of test 9 and "
        f"must read exactly zero at a fixed iteration count too"
    )
    assert deltas[-1] > ABOVE_FLOOR, (
        f"fixture: the smallest mu must stay above the null floor, {deltas}"
    )
    assert all(a > b for a, b in zip(deltas, deltas[1:])), (
        f"not monotone in mu: {deltas} for {mus}"
    )
    assert deltas[-1] <= 0.1 * deltas[0], (
        f"delta_r_free must fall by at least a decade across three decades of mu: "
        f"{deltas} for {mus}. The measured exponent is ~0.67, not 1: at small mu the "
        f"solve converges further within the fixed iteration count, and the larger x "
        f"partly offsets the weaker coupling"
    )


def test_14_phi_rf_seed_and_degree_are_frozen_across_the_solve():
    g = _gctx(14)
    g = g._replace(mu_graph=_f64(_mu_balanced(g, 1401)))
    phi_before = np.array(g.phi, copy=True)
    seed_before = g.rf_seed
    d_before = np.array(gmrf.degree(g.phi, g.rf_seed, g.m, g.kernel, g.chunk), copy=True)
    x, _ = _solve(g, 0.0, MAX_ITER)
    assert float(jnp.max(jnp.abs(x))) > 0.0, "fixture: the solve must do something"
    d_after = np.asarray(gmrf.degree(g.phi, g.rf_seed, g.m, g.kernel, g.chunk))
    assert g.rf_seed == seed_before, f"rf_seed changed: {seed_before} -> {g.rf_seed}"
    assert np.asarray(g.phi).dtype == phi_before.dtype, "phi dtype changed"
    assert np.asarray(g.phi).shape == phi_before.shape, "phi shape changed"
    assert np.array_equal(np.asarray(g.phi), phi_before), (
        f"phi moved by {_rel(g.phi, phi_before):.3e} across {MAX_ITER} steps. {FREEZE_MSG}"
    )
    assert d_after.dtype == d_before.dtype and d_after.shape == d_before.shape, "d changed type"
    assert np.array_equal(d_after, d_before), (
        f"the degree moved by {_rel(d_after, d_before):.3e} across {MAX_ITER} steps, so it "
        f"is not reproducible from (phi, rf_seed, m, kernel, chunk) alone. {FREEZE_MSG}"
    )


def test_15_the_solver_does_not_mutate_its_inputs():
    g = _gctx(15)
    g = g._replace(mu_graph=_f64(_mu_balanced(g, 1501)))
    x0 = _zeros()
    arrays = {f: getattr(g, f) for f in g._fields if isinstance(getattr(g, f), jax.Array)}
    before = {f: np.array(a, copy=True) for f, a in arrays.items()}
    x0_before = np.array(x0, copy=True)
    x, _ = estimate.solve(x0, g, 0.0, MAX_ITER)
    assert float(jnp.max(jnp.abs(x))) > 0.0, "fixture: the solve must do something"
    assert np.array_equal(np.asarray(x0), x0_before), (
        f"x0 changed across the solve by {_rel(x0, x0_before):.3e}; solve must not donate "
        f"or write through its starting field"
    )
    for f, ref in before.items():
        assert np.array_equal(np.asarray(getattr(g, f)), ref), (
            f"ctx.{f} changed across the solve by {_rel(getattr(g, f), ref):.3e}"
        )


def test_16_same_seed_and_config_reproduce_delta_r_free():
    g = _gctx(16)
    g = g._replace(mu_graph=_f64(_mu_balanced(g, 1601)))
    eps = _eps_floor(g)
    a, xa, ta = _drf(g, eps)
    b, xb, tb = _drf(g, eps)
    assert abs(a) > ABOVE_FLOOR, f"fixture: the reading must be non-trivial, got {a:.3e}"
    assert int(ta.t_star) == int(tb.t_star), f"t* differs between runs: {ta.t_star}, {tb.t_star}"
    assert abs(a - b) <= TOL_NULL * max(abs(a), abs(b)), (
        f"delta_r_free differs between two identical runs: {a!r} vs {b!r}, relative "
        f"{abs(a - b) / max(abs(a), abs(b)):.3e}"
    )
    assert _rel(xa, xb) <= TOL_NULL, f"the refined fields differ by {_rel(xa, xb):.3e}"


def test_17_rf_seed_selects_the_graph_and_reproduces_it():
    g = _gctx(17)
    other = g._replace(rf_seed=g.rf_seed + 1)
    x = _normal(1701, SHAPE)
    a = _graph(x, g, 1.0)
    again = _graph(x, g, 1.0)
    b = _graph(x, other, 1.0)
    assert float(jnp.max(jnp.abs(a))) > 0.0, "fixture: the graph must be non-trivial"
    assert np.array_equal(np.asarray(a), np.asarray(again)), (
        f"the same rf_seed gave two different operators, differing by {_rel(a, again):.3e}; "
        f"the graph must be bitwise reproducible from the stored seed"
    )
    assert np.array_equal(
        np.asarray(gmrf.rf_block(g.phi, g.rf_seed, g.m, g.kernel)),
        np.asarray(gmrf.rf_block(g.phi, g.rf_seed, g.m, g.kernel)),
    ), "rf_block must be bitwise reproducible from the stored seed"
    assert _rel(a, b) > NONZERO, (
        f"changing rf_seed moved the operator by only {_rel(a, b):.3e}; S6 requires the "
        f"seed to be stored, never derived from the shape, precisely because it selects "
        f"the graph"
    )


def test_18_preconditioned_ncg_reaches_the_noise_floor_from_zero():
    g = _gctx(18)
    g = g._replace(mu_graph=_f64(_mu_balanced(g, 1801)))
    eps = _reachable_floor(g)
    r0 = _r_work0(g)
    assert eps < r0, f"fixture: the floor {eps:.4f} must lie below R_work at x0 = 0, {r0:.4f}"
    x, trace = _solve(g, eps, MAX_ITER_LONG)
    t = int(trace.t_star)
    assert bool(trace.at_floor), (
        f"the run ended at the iteration cap {MAX_ITER_LONG}, not at the floor; R_work "
        f"reached {float(trace.r_work[t]):.4f} against eps_floor {eps:.4f}"
    )
    assert 0 < t < MAX_ITER_LONG, f"t* = {t} outside (0, {MAX_ITER_LONG})"
    assert float(trace.r_work[t]) <= eps, (
        f"R_work at t* = {float(trace.r_work[t]):.6f} is above eps_floor {eps:.6f}"
    )
    assert all(float(trace.r_work[i]) > eps for i in range(t)), (
        f"t* must be the first step at or below the floor, not step {t}"
    )
    assert float(jnp.sum(x ** 2)) > 0.0, "x0 = 0 must have moved"
    assert _rel(x, _pi(x, g)) <= TOL, (
        f"the refined x left the range of Pi by {_rel(x, _pi(x, g)):.3e}; S5 constrains "
        f"the variable to Pi x = x"
    )
    assert float(trace.gnorm[t]) < float(trace.gnorm[0]), (
        f"||g|| did not fall: {float(trace.gnorm[0]):.3e} -> {float(trace.gnorm[t]):.3e}"
    )
    assert all(np.isfinite(np.asarray(trace.j)[: t + 1])), "J must stay finite over the run"


def test_19_the_null_holds_at_each_configuration_own_t_star():
    g = _gctx(19)
    _assert_null_fixture(g)
    unit = _mu_balanced(g, 1901)
    mus = tuple(unit * f for f in (0.0, 0.1, 1.0, 10.0))
    eps = _reachable_floor(g._replace(mu_graph=_f64(unit * 10.0)))
    stars, floors = [], []
    for mu in mus:
        _, trace = _solve(g._replace(mu_graph=_f64(mu)), eps, MAX_ITER_LONG)
        stars.append(int(trace.t_star))
        floors.append(bool(trace.at_floor))
    reached = [t for t, f in zip(stars, floors) if f]
    assert len(reached) >= 2, (
        f"fixture: at least two configurations must reach the floor within "
        f"{MAX_ITER_LONG} steps; t* {stars}, at_floor {floors}"
    )
    assert reached == sorted(reached), (
        f"t* fell as mu grew: {stars} for {mus}. A stronger prior slows the approach to "
        f"the floor, so t* must not decrease"
    )
    assert len(set(reached)) > 1, (
        f"fixture: t* must move with mu, otherwise this test is a fixed iteration count "
        f"under another name; got {stars}"
    )
    off = g._replace(mu_graph=_f64(0.0))
    for t in sorted(set(stars)):
        x, trace = estimate.solve(_zeros(), off, 0.0, t)
        assert int(trace.t_star) == t, f"a cap of {t} must give t* = {t}, got {int(trace.t_star)}"
        d = float(estimate.delta_r_free(x, off))
        work = _assert_moved(x, off, f"t* = {t}")
        free = _band_max(x, off.mask_free)
        assert free <= TOL_NULL * work, f"t* = {t}: free/work = {free / work:.3e}"
        assert abs(d) <= TOL_NULL, (
            f"the mu = 0 null reads {d:.6e} at t* = {t}. The null is a property of the "
            f"precision being a multiplier, not of the step count, so it must hold at "
            f"every configuration's own stopping step"
        )


def test_20_condition_number_is_finite_and_reported_for_every_mu():
    g = _gctx(20)
    unit = _mu_balanced(g, 2001)
    mus = (0.0,) + tuple(unit * f for f in (1e-2, 1e-1, 1e0, 1e1, 1e2))
    reported = []
    for i, mu in enumerate(mus):
        gm = g._replace(mu_graph=_f64(mu))
        evals = np.sort(np.asarray(
            gmrf.graph_spectrum(lambda v: estimate.P_apply(v, gm), N_EIG, _key(2010 + i), SHAPE)
        ))
        assert np.all(np.isfinite(evals)), f"mu = {mu:.3e}: Lanczos returned {evals}"
        top = float(evals[-1])
        assert top > 0.0, f"mu = {mu:.3e}: the largest Ritz value must be positive"
        keep = evals[evals > KAPPA_ZERO * top]
        dropped = evals[evals <= KAPPA_ZERO * top]
        assert keep.size > 1, (
            f"mu = {mu:.3e}: fewer than two Ritz values above {KAPPA_ZERO:.0e} of the top; "
            f"{evals}"
        )
        if dropped.size:
            assert float(keep.min()) > 1e3 * max(float(dropped.max()), 0.0), (
                f"mu = {mu:.3e}: no clean gap between the null space of Pi and the spectrum "
                f"on the band; kept min {float(keep.min()):.3e}, dropped max "
                f"{float(dropped.max()):.3e}"
            )
        kappa = float(keep.max() / keep.min())
        assert np.isfinite(kappa) and kappa >= 1.0, f"mu = {mu:.3e}: kappa(P) = {kappa}"
        reported.append(kappa)
    print("test 20 kappa(P): " + ", ".join(f"mu {m:.3e} -> {k:.4e}" for m, k in zip(mus, reported)))
    assert reported[-1] > reported[0], (
        f"kappa(P) did not grow across four decades of mu: {reported}. The graph term is "
        f"unpreconditioned, so conditioning must degrade as mu grows; a flat kappa means "
        f"the coupling is not reaching P"
    )


def _graph_apply(x, phi, seed, m, mu, kernel, chunk):
    return estimate.P_graph(x, phi, seed, m, mu, kernel, chunk)


def test_21_memory_report_matches_the_peak_and_refuses_an_oversized_config():
    g = _gctx(21)
    n, d = int(g.phi.shape[0]), int(g.phi.shape[1])
    report = estimate.memory_report(n, d, g.m, g.chunk, 10 ** 9)
    assert (report.n, report.d, report.m, report.chunk) == (n, d, g.m, g.chunk), (
        f"the report must restate N, d, m and the chunk size, got {report}"
    )
    assert report.peak_bytes > 0, f"peak_bytes = {report.peak_bytes}"
    doubled = estimate.memory_report(n, d, g.m, 2 * g.chunk, 10 ** 9)
    cost = doubled.peak_bytes - report.peak_bytes
    assert abs(cost - 8 * g.chunk * g.m) <= 8 * g.m, (
        f"doubling the chunk cost {cost} bytes, not the {8 * g.chunk * g.m} that one more "
        f"chunk of Z takes; S6 forbids materialising Z, so the footprint must scale with "
        f"chunk*m, never N*m"
    )
    with pytest.raises(ValueError) as caught:
        estimate.memory_report(n, d, g.m, g.chunk, report.peak_bytes - 1)
    assert str(report.peak_bytes) in str(caught.value), (
        f"the refusal must carry the footprint it refused: {caught.value}"
    )
    fn = jax.jit(partial(
        _graph_apply, seed=g.rf_seed, m=g.m, mu=1.0, kernel=g.kernel, chunk=g.chunk
    ))
    x = _pi(_normal(2101, SHAPE), g)
    analysis = fn.lower(x, g.phi).compile().memory_analysis()
    if analysis is None:
        pytest.skip("memory_analysis is unavailable on this backend")
    measured = (
        int(analysis.argument_size_in_bytes)
        + int(analysis.output_size_in_bytes)
        + int(analysis.temp_size_in_bytes)
        - int(getattr(analysis, "alias_size_in_bytes", 0))
    )
    assert measured > 0, f"the compiled matvec reported {measured} bytes"
    assert abs(report.peak_bytes - measured) <= TOL_MEMORY * measured, (
        f"reported peak {report.peak_bytes} vs measured {measured}, "
        f"{abs(report.peak_bytes - measured) / measured:.1%} apart; a predictable refusal "
        f"is worth nothing if the prediction is wrong"
    )


def _phase_b_fixture(seed):
    g = _gctx(seed)
    params = _cnn_params(10 * seed)
    phi = gmrf.embed_cnn(params, g.rho0)
    assert phi.shape == (_n(SHAPE), D_OUT), f"embed_cnn must give (N, d), got {phi.shape}"
    g = g._replace(mu_graph=_f64(_mu_balanced(_phase_b_ctx(g, params), 10 * seed + 1)))
    return g, params


def test_22_unrolled_hypergradient_matches_central_finite_differences():
    g, params = _phase_b_fixture(22)
    loss = _phase_b_loss(params, g, 0.0, N_UNROLL)
    assert np.isfinite(float(loss)) and abs(float(loss)) > ABOVE_FLOOR, (
        f"fixture: the Phase B loss must be finite and non-trivial, got {float(loss):.3e}"
    )
    ana = estimate.hypergrad_unrolled(params, g, 0.0, N_UNROLL, N_UNROLL)
    gw = np.asarray(ana["w"])
    assert np.all(np.isfinite(gw)), "the unrolled hypergradient is not finite"
    order = np.argsort(np.abs(gw).ravel())[::-1][:5]
    assert float(np.abs(gw).ravel()[order[-1]]) > 0.0, (
        "fixture: at least five parameters must carry a non-zero hypergradient"
    )
    for flat in order:
        at = np.unravel_index(int(flat), gw.shape)
        bump = np.zeros_like(gw)
        bump[at] = FD_H
        plus = float(_phase_b_loss(
            {**params, "w": params["w"] + bump}, g, 0.0, N_UNROLL))
        minus = float(_phase_b_loss(
            {**params, "w": params["w"] - bump}, g, 0.0, N_UNROLL))
        fd = (plus - minus) / (2.0 * FD_H)
        span = max(abs(fd), abs(float(gw[at])))
        assert abs(fd - float(gw[at])) <= TOL_FD * span, (
            f"w{at}: unrolled {float(gw[at]):.9e} vs central difference {fd:.9e} at "
            f"h = {FD_H:.0e}, relative {abs(fd - float(gw[at])) / span:.3e}. {UNROLL_MSG}"
        )


def test_23_implicit_hypergradient_matches_the_unrolled_one():
    g, params = _phase_b_fixture(23)
    eps = _reachable_floor(_phase_b_ctx(g, params), N_CONVERGED)
    x, trace = _solve(_phase_b_ctx(g, params), eps, N_CONVERGED)
    t = int(trace.t_star)
    assert bool(trace.at_floor), (
        f"fixture: the implicit form is the KKT system of min 0.5 x'Px s.t. E_work = "
        f"eps_floor, so the solve must actually reach the floor; it ended at the cap with "
        f"R_work {float(trace.r_work[t]):.4f} against {eps:.4f}"
    )
    assert abs(float(trace.r_work[t]) - eps) <= 0.05 * eps, (
        f"fixture: the constraint E_work = eps_floor must be met at t*; R_work is "
        f"{float(trace.r_work[t]):.4f} against {eps:.4f}. S9 reformulates precisely "
        f"because the floor stop is not a stationary point of the unconstrained problem, "
        f"so requiring ||g|| -> 0 here would contradict the thing being tested"
    )
    unrolled = estimate.hypergrad_unrolled(params, g, eps, N_CONVERGED, N_CONVERGED)
    implicit = estimate.hypergrad_implicit(params, g, eps, N_CONVERGED)
    for name in ("w", "b"):
        a = np.asarray(unrolled[name])
        b = np.asarray(implicit[name])
        assert a.shape == b.shape, f"{name}: shapes differ, {a.shape} vs {b.shape}"
        assert np.all(np.isfinite(b)), f"{name}: the implicit hypergradient is not finite"
        assert float(np.max(np.abs(a))) > 0.0, f"fixture: {name} hypergradient must be non-zero"
        assert _rel(a, b) <= TOL_IMPLICIT, (
            f"{name}: implicit differs from unrolled by {_rel(a, b):.3e} relative. The "
            f"noise-floor stop is deliberately not a stationary point, so the implicit "
            f"form must differentiate the KKT system of the constrained problem of S9, not "
            f"the stopped iteration"
        )


NCS_SHAPE = (16, 16, 16)
NCS_BLOCK = (1, 8)
NCS_T = (8, 8, 8)
K_NCS = 8
TOP_FRACTION = 0.01


def _ncs_rho0(seed, with_copy):
    lo, hi = NCS_BLOCK
    rho = np.asarray(_normal(seed, NCS_SHAPE), dtype=np.float64).copy()
    motif = 8.0 * np.asarray(_normal(seed + 1, (hi - lo, hi - lo, hi - lo)), dtype=np.float64)
    rho[lo:hi, lo:hi, lo:hi] = motif
    if with_copy:
        t0, t1, t2 = NCS_T
        rho[lo + t0:hi + t0, lo + t1:hi + t1, lo + t2:hi + t2] = motif
    return jnp.asarray(rho)


def _ncs_matched_fraction(rho0):
    phi = gmrf.concat_features(gmrf.patch_features(rho0, PATCH, 1), whiten=True)
    idx, w = gmrf.knn_graph(phi, K_NCS)
    idx = np.asarray(idx)
    w = np.asarray(w)
    n = idx.shape[0]
    rows = np.repeat(np.arange(n), idx.shape[1])
    top = max(1, int(round(TOP_FRACTION * rows.size)))
    order = np.argsort(w.ravel())[::-1][:top]
    src = np.stack(np.unravel_index(rows[order], NCS_SHAPE), axis=1)
    dst = np.stack(np.unravel_index(idx.ravel()[order], NCS_SHAPE), axis=1)
    shape = np.asarray(NCS_SHAPE)
    disp = (dst - src) % shape
    err = np.minimum((disp - np.asarray(NCS_T)) % shape, (np.asarray(NCS_T) - disp) % shape)
    return float(np.mean(np.max(err, axis=1) <= 1)), top


def test_24_ncs_displacement_fraction_among_the_top_graph_weights():
    ncs, top = _ncs_matched_fraction(_ncs_rho0(2401, True))
    control, _ = _ncs_matched_fraction(_ncs_rho0(2401, False))
    chance = 27.0 / _n(NCS_SHAPE)
    print(
        f"test 24 top-{TOP_FRACTION:.0%} ({top} edges) matching the known (I, {NCS_T}) "
        f"within one grid spacing: NCS {ncs:.4f}, no-copy control {control:.4f}, "
        f"chance {chance:.4f}. {NCS_MSG}"
    )
    assert 0.0 <= ncs <= 1.0 and np.isfinite(ncs), f"the reported fraction is {ncs}"
    assert control <= 10.0 * chance, (
        f"the control map has no second copy yet {control:.4f} of its top weights match "
        f"the displacement, against a chance rate of {chance:.4f}; the measurement is "
        f"picking up something other than the NCS"
    )
    assert ncs > 0.5, (
        f"only {ncs:.4f} of the top weights connect a voxel to its NCS mate, against "
        f"{control:.4f} in the control. The graph is not finding the copies"
    )
