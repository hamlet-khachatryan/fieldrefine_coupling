from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import core

from . import gmrf

GCtx = NamedTuple("GCtx", [
    ("rho0", jax.Array), ("F0", jax.Array), ("amp", jax.Array), ("wgt", jax.Array),
    ("mult", jax.Array), ("mask_work", jax.Array), ("mask_free", jax.Array),
    ("mask_obs", jax.Array), ("sqrtS", jax.Array), ("sqrtS_sigma", jax.Array),
    ("sym_idx", jax.Array), ("sym_phase", jax.Array), ("sym_conj", jax.Array),
    ("s_min", jax.Array), ("rho_floor", jax.Array), ("mu", jax.Array),
    ("inv_S", jax.Array), ("phi", jax.Array), ("rf_seed", int),
    ("mu_graph", jax.Array), ("m", int), ("kernel", str), ("chunk", int),
    ("knn", object),
])
GTrace = NamedTuple("GTrace", [
    ("j", jax.Array), ("e_work", jax.Array), ("e_pos", jax.Array), ("x_p_x", jax.Array),
    ("r_work", jax.Array), ("r_free", jax.Array), ("gnorm", jax.Array),
    ("t_star", jax.Array), ("at_floor", jax.Array),
])
Memory = NamedTuple("Memory", [
    ("n", int), ("d", int), ("m", int), ("chunk", int), ("peak_bytes", int),
])


def _core(ctx):
    return core.Ctx(**{f: getattr(ctx, f) for f in core.Ctx._fields})


def P_stat(x, inv_S):
    """Apply the stationary part of the precision.

    A positive Fourier multiplier, so it is self-adjoint, positive definite on the band,
    and free to invert -- which is what makes it the preconditioner as well. Being a
    multiplier it cannot move power between mask_work and mask_free, which is the whole
    content of the mu = 0 null.
    """
    return core.apply_mult(x, inv_S)


def P_graph(x, phi, seed, m, mu, kernel, chunk):
    """Apply the coupling part of the precision, ``mu (D - Z Z^T)``.

    The degree is recomputed here rather than cached, from the same two-pass routine the
    matvec uses. That costs a second pass and buys two properties the tests rest on: the
    graph is reproducible from ``(phi, rf_seed, m, kernel, chunk)`` alone at any point in
    the solve, and ``d * 1`` cancels ``Z(Z^T 1)`` bitwise rather than to round-off.
    """
    f = x.ravel()
    d = gmrf.degree(phi, seed, m, kernel, chunk)
    return (mu * (d * f - gmrf.apply_zzt(phi, seed, m, kernel, chunk, f))).reshape(x.shape)


def _knn_apply(f, idx, w):
    wx = jnp.sum(w * f[idx], axis=1)
    wtx = jnp.zeros_like(f).at[idx.ravel()].add((w * f[:, None]).ravel())
    # a k-nearest-neighbour relation is not symmetric
    return 0.5 * (wx + wtx)


def P_graph_knn(x, idx, w, mu):
    """Apply the coupling part from an explicit sparse graph.

    W is symmetrised before the degree is taken, so D - W is symmetric and PSD; taking
    the degree from the unsymmetrised relation gives an operator that is neither, and it
    diverges rather than failing cleanly.
    """
    f = x.ravel()
    d = _knn_apply(jnp.ones_like(f), idx, w)
    return (mu * (d * f - _knn_apply(f, idx, w))).reshape(x.shape)


def P_apply(x, ctx):
    """Apply the full precision, projected onto the band-limited symmetric subspace.

    ``Pi(P_stat + P_graph)`` is self-adjoint on the range of Pi, which is where S5 puts
    the variable; it is not self-adjoint off that subspace, since Pi does not commute
    with the graph term.
    """
    if ctx.knn is None:
        g = P_graph(x, ctx.phi, ctx.rf_seed, ctx.m, ctx.mu_graph, ctx.kernel, ctx.chunk)
    else:
        g = P_graph_knn(x, ctx.knn[0], ctx.knn[1], ctx.mu_graph)
    return core.pi_real(P_stat(x, ctx.inv_S) + g, _core(ctx))


def _precondition(r, ctx):
    return core.pi_real(core.apply_mult(r, 1.0 / ctx.inv_S), _core(ctx))


def objective(x, ctx):
    """Evaluate the objective of S5 and its terms.

    The scale is fitted on ``mask_work`` only. Fitting it across ``mask_obs`` is the
    commonest way to couple work to free: k then carries free amplitudes into every
    residual, and the null reads a small non-zero value for a reason invisible in the
    gradient.
    """
    cv = _core(ctx)
    rho = ctx.rho0 + x
    F = core.sf(rho, cv)
    k = core.fit_scale(F, cv, ctx.mask_work)
    ew = core.e_work(F, cv, k)
    ep = core.e_pos(rho, cv)
    xpx = 0.5 * jnp.sum(x * P_apply(x, ctx))
    return ew + ctx.mu * ep + xpx, (ew, ep, xpx, k, F)


def grad(x, ctx):
    """Analytic gradient of S5.

    ``resid_coeff`` carries ``wgt`` but not ``mult``: core's ``c2r`` is already the
    adjoint of ``r2c`` under the multiplicity inner product, so applying it here too
    would double every coefficient off the self-conjugate planes. The data part is
    supported on ``mask_work``, so with a multiplier P a free coefficient that starts at
    zero never receives a gradient.
    """
    cv = _core(ctx)
    rho = ctx.rho0 + x
    F = core.sf(rho, cv)
    k = core.fit_scale(F, cv, ctx.mask_work)
    g_rho = core.c2r(core.resid_coeff(F, cv, k), rho.shape) + ctx.mu * core.neg_part(rho, cv)
    return core.pi_real(g_rho, cv) + P_apply(x, ctx)


def _readout(x, ctx):
    F = core.sf(ctx.rho0 + x, _core(ctx))
    return F, core.fit_scale(F, _core(ctx), ctx.mask_work)


def delta_r_free(x, ctx):
    """Change in R_free between the starting model and the refined field.

    One work-fitted scale for both R factors: refitting per state would move R_free
    through the scale alone and report a non-zero result where nothing coupled.
    """
    cv = _core(ctx)
    F, k = _readout(x, ctx)
    return core.r_factor(ctx.F0, cv, k, ctx.mask_free) - core.r_factor(F, cv, k, ctx.mask_free)


def eta_c(x, ctx):
    """Ratio of the change in R_free to the change in R_work, on one scale."""
    cv = _core(ctx)
    F, k = _readout(x, ctx)
    dw = core.r_factor(ctx.F0, cv, k, ctx.mask_work) - core.r_factor(F, cv, k, ctx.mask_work)
    return delta_r_free(x, ctx) / dw


def _row(x, ctx):
    cv = _core(ctx)
    j, (ew, ep, xpx, k, F) = objective(x, ctx)
    return (
        j, ew, ep, xpx,
        core.r_factor(F, cv, k, ctx.mask_work),
        core.r_factor(F, cv, k, ctx.mask_free),
        jnp.sqrt(jnp.sum(grad(x, ctx) ** 2)),
    )


def solve(x0, ctx, eps_floor, max_iter):
    """Preconditioned nonlinear conjugate gradient from x0, stopped at the noise floor.

    Polak-Ribiere with beta clipped at zero, and a restart whenever the resulting
    direction is not a descent direction -- without that test a large beta builds a
    direction pointing along the unpreconditioned graph term, where the curvature is
    negative. The step length is an exact line search on the local quadratic model,
    ``<r,p>/<p,Hp>``, with the curvature floored at ``<p,Pp>``: P is positive definite
    but the Hessian of the amplitude residual is not, and an unfloored negative
    curvature gives a zero step, which is a fixed point the iteration never leaves.
    Both guards are ``where``, not branches, so the iteration stays differentiable for
    S9.

    Once R_work reaches ``eps_floor`` the update is frozen rather than the loop broken,
    so the returned field is the one at t* while the trace still runs to the cap. R_free
    is computed for the trace and for nothing else: it enters neither the step, the step
    length, nor the stop.
    """
    def body(carry, _):
        x, r, z, p, frozen, t, t_star = carry
        h = jax.jvp(lambda v: grad(v, ctx), (x,), (p,))[1]
        # P is positive definite; the |F| Hessian is not
        model = jnp.maximum(jnp.sum(p * h), jnp.sum(p * P_apply(p, ctx)))
        ok = model > 0.0
        alpha = jnp.where(ok, jnp.sum(r * p) / jnp.where(ok, model, 1.0), 0.0)
        x_new = jnp.where(frozen, x, x + alpha * p)
        r_new = jnp.where(frozen, r, -grad(x_new, ctx))
        z_new = jnp.where(frozen, z, _precondition(r_new, ctx))
        denom = jnp.sum(z * r)
        beta = jnp.maximum(
            0.0, jnp.sum(z_new * (r_new - r)) / jnp.where(denom != 0.0, denom, 1.0)
        )
        p_try = z_new + beta * p
        p_new = jnp.where(jnp.sum(r_new * p_try) > 0.0, p_try, z_new)
        p_new = jnp.where(frozen, p, p_new)
        row = _row(x_new, ctx)
        t = t + 1
        hit = row[4] <= eps_floor
        t_star = jnp.where(frozen, t_star, jnp.where(hit, t, t_star))
        return (x_new, r_new, z_new, p_new, frozen | hit, t, t_star), row

    r0 = -grad(x0, ctx)
    z0 = _precondition(r0, ctx)
    row0 = _row(x0, ctx)
    done0 = row0[4] <= eps_floor
    start = (x0, r0, z0, z0, done0, jnp.asarray(0), jnp.where(done0, 0, max_iter))
    (x, _, _, _, frozen, _, t_star), rows = jax.lax.scan(body, start, None, length=max_iter)
    trace = GTrace(
        *[jnp.concatenate([jnp.asarray(a)[None], b]) for a, b in zip(row0, rows)],
        t_star=t_star, at_floor=frozen,
    )
    return x, trace


def memory_report(n, d, m, chunk, budget):
    """Report the footprint of the graph matvec and refuse a configuration over budget.

    Counted: the stored phi, the padded copy the chunked reshape makes when the chunk
    does not divide N, one chunk of Z, the accumulator s, and four N-length fields (the
    input, the padded input, the keep mask, and the stacked scan output). A stored Z
    would be ``n * m * 8`` and is what this exists to avoid -- 1.3 GB at n = 1e6,
    m = 128, and 130 GB at 512^3. A predictable refusal beats an OOM at hour three.

    Calibrated against ``memory_analysis`` of the compiled matvec over five
    configurations spanning N, d, m and chunk; worst observed error 6%. The padded copy
    is conditional because XLA aliases the reshape when the chunk divides N exactly, and
    counting it unconditionally overestimated by 20% in that case.
    """
    pad = (-n) % chunk
    peak = int(8 * (n * d + (n + pad) * d * (pad > 0) + chunk * m + m + 4 * n))
    if peak > budget:
        raise ValueError(
            f"graph matvec needs {peak} bytes (N={n}, d={d}, m={m}, chunk={chunk}) "
            f"but the budget is {budget}"
        )
    return Memory(int(n), int(d), int(m), int(chunk), peak)


def hypergrad_unrolled(params, ctx, eps_floor, max_iter, window):
    """Gradient of delta_r_free with respect to psi, by differentiating the last steps.

    ``stop_gradient`` on the state entering the window, so memory is bounded by
    ``window`` rather than by ``max_iter``. Always valid; with ``window == max_iter`` it
    is the exact derivative, which is the case test 22 pins against finite differences.
    """
    def loss(p):
        gg = ctx._replace(phi=gmrf.embed_cnn(p, ctx.rho0))
        x0 = jnp.zeros(ctx.rho0.shape, dtype=jnp.float64)
        if window < max_iter:
            held, _ = solve(
                x0, gg._replace(phi=jax.lax.stop_gradient(gg.phi)),
                eps_floor, max_iter - window,
            )
            x0 = jax.lax.stop_gradient(held)
        x, _ = solve(x0, gg, eps_floor, min(window, max_iter))
        return delta_r_free(x, gg)

    return jax.grad(loss)(params)


def _e_work(x, ctx):
    cv = _core(ctx)
    F = core.sf(ctx.rho0 + x, cv)
    return core.e_work(F, cv, core.fit_scale(F, cv, ctx.mask_work))


def _tangent_solve(apply_A, rhs, g, n_iter):
    # conjugate gradient confined to the tangent space of the constraint
    def project(v):
        return v - g * (jnp.sum(g * v) / jnp.sum(g * g))

    x = jnp.zeros_like(rhs)
    r = project(rhs)
    p, rs = r, jnp.sum(r * r)
    for _ in range(n_iter):
        ap = project(apply_A(p))
        denom = jnp.sum(p * ap)
        a = jnp.where(denom > 0.0, rs / jnp.where(denom > 0.0, denom, 1.0), 0.0)
        x = x + a * p
        r = r - a * ap
        rs_new = jnp.sum(r * r)
        p = r + (rs_new / jnp.where(rs > 0.0, rs, 1.0)) * p
        rs = rs_new
    return x


def hypergrad_implicit(params, ctx, eps_floor, max_iter, n_cg=60):
    """Gradient of delta_r_free with respect to psi, by the implicit function theorem.

    The noise-floor stop is deliberately not a stationary point, so the IFT is applied to
    the reformulation of S9, ``min 0.5 x'Px s.t. E_work(x) = eps_floor``, whose solution
    *is* stationary for the Lagrangian. Differentiating the KKT system gives

        [H  g] [dx    ]   [-(dP/dpsi) x]
        [g' 0] [dlambda] = [0          ]

    with ``H = P + lambda d2E`` and ``g = dE/dx``. The system is symmetric, so rather than
    solving it once per parameter the adjoint is solved once: ``u`` satisfying the same
    system with the readout gradient on the right, after which the hypergradient is
    ``-u . (dP/dpsi) x``, one matrix-free CG regardless of how many parameters psi has.

    CG is confined to the tangent space of the constraint, which is what the bordered row
    expresses; that keeps the operator positive definite there and avoids the indefinite
    bordered system MINRES would otherwise be needed for.
    """
    phi = gmrf.embed_cnn(params, ctx.rho0)
    gg = ctx._replace(phi=phi)
    x, _ = solve(jnp.zeros(ctx.rho0.shape, dtype=jnp.float64), gg, eps_floor, max_iter)
    x = jax.lax.stop_gradient(x)

    g = jax.grad(_e_work)(x, gg)
    lam = -jnp.sum(x * P_apply(x, gg)) / jnp.where(
        jnp.sum(g * x) != 0.0, jnp.sum(g * x), 1.0
    )

    def apply_H(v):
        d2e = jax.jvp(lambda w: jax.grad(_e_work)(w, gg), (x,), (v,))[1]
        return P_apply(v, gg) + lam * d2e

    v = jax.grad(lambda w: delta_r_free(w, gg))(x)
    u = _tangent_solve(apply_H, v, g, n_cg)

    def coupling(p):
        return jnp.sum(jax.lax.stop_gradient(u) * P_apply(jax.lax.stop_gradient(x),
                                                          ctx._replace(phi=gmrf.embed_cnn(p, ctx.rho0))))

    return jax.tree_util.tree_map(lambda a: -a, jax.grad(coupling)(params))
