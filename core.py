from functools import partial
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

Ctx = NamedTuple("Ctx", [
    ("rho0", jax.Array), ("F0", jax.Array), ("amp", jax.Array), ("wgt", jax.Array),
    ("mult", jax.Array), ("mask_work", jax.Array), ("mask_free", jax.Array),
    ("mask_obs", jax.Array), ("sqrtS", jax.Array), ("sqrtS_sigma", jax.Array),
    ("sym_idx", jax.Array), ("sym_phase", jax.Array), ("sym_conj", jax.Array),
    ("s_min", jax.Array),
    ("rho_floor", jax.Array), ("mu", jax.Array),
])
State = NamedTuple("State", [("z", jax.Array), ("u", jax.Array)])
Trace = NamedTuple("Trace", [
    ("j", jax.Array), ("e_work", jax.Array), ("r_work", jax.Array), ("r_free", jax.Array),
])
Terms = NamedTuple("Terms", [
    ("e_work", jax.Array), ("e_pos", jax.Array), ("e_z", jax.Array), ("e_u", jax.Array),
    ("k", jax.Array),
])
Knobs = NamedTuple("Knobs", [("c", float), ("q_sigma", float), ("mu", float)])


@jax.jit
def r2c(x):
    """Transform a real map to its half spectrum.

    Orthonormal, so the transform is unitary and Parseval holds without a scale factor
    -- but only once the Friedel multiplicity is carried, since the half spectrum omits
    one member of every pair off the self-conjugate planes.
    """
    return jnp.fft.rfftn(x, axes=(0, 1, 2), norm="ortho")


@partial(jax.jit, static_argnums=(1,))
def c2r(F, shape):
    """Transform a half spectrum back to a real map.

    The adjoint of ``r2c`` under the multiplicity-weighted inner product, with constant
    1. That is why ``resid_coeff`` must not carry ``mult`` as well: the multiplicity is
    already in this adjoint, and applying it twice would double every coefficient off
    the self-conjugate planes.
    """
    return jnp.fft.irfftn(F, s=shape, axes=(0, 1, 2), norm="ortho")


@jax.jit
def apply_mult(x, s):
    """Apply a reciprocal-space multiplier to a real map.

    Self-adjoint for real, Friedel-symmetric ``s``, and composes multiplicatively, so
    ``apply_mult(apply_mult(x, s), t)`` equals ``apply_mult(x, s * t)``. This is the
    only way a spectral shape enters the forward model.
    """
    return c2r(r2c(x) * s, x.shape)


@jax.jit
def project_band(F, mask_obs):
    """Keep only the measured coefficients.

    Idempotent and self-adjoint because the mask is a 0/1 indicator. It commutes with
    ``project_sym`` exactly when the mask is symmetry-invariant, which is why the
    adapter expands every reflection over its full orbit.
    """
    return F * mask_obs


@jax.jit
def project_sym(F, sym_idx, sym_phase, sym_conj):
    """Average a spectrum over the space group.

    For each operator the value at ``M h`` is gathered, conjugated where the fold into
    the stored half grid requires it, and multiplied by the operator's phase. The mean
    over operators is the projection onto the symmetric subspace: idempotent always,
    and self-adjoint under the multiplicity-weighted inner product exactly when the
    operator set is closed under inverse, since the adjoint of one operator is the
    operator of the inverse element.
    """
    g = F.ravel()[sym_idx]
    return jnp.mean(jnp.where(sym_conj, jnp.conj(g), g) * sym_phase, axis=0)


@jax.jit
def pi_real(x, ctx):
    """Project a real map onto the band-limited, symmetric subspace.

    Band projection then symmetrisation, back in real space. Idempotent and
    self-adjoint provided the two projectors commute, which the adapter guarantees.
    Self-adjointness is what lets ``grad`` push the real-space gradient back through
    the same projector instead of a transpose of it.
    """
    F = project_sym(
        project_band(r2c(x), ctx.mask_obs), ctx.sym_idx, ctx.sym_phase, ctx.sym_conj
    )
    return c2r(F, x.shape)


def sigma_spectrum(c, q_sigma, ctx):
    """Build the spectral shape of the log-amplitude field.

    A flat indicator on ``|q| <= q_sigma``, scaled so that ``apply_mult(u, .)`` has
    variance ``c**2`` for white ``u``; with ``Var = sum(mult * S) / N`` that fixes the
    level uniquely. ``c = 0`` returns exactly zero, hence a constant sigma -- the only
    property the null argument needs.
    """
    n0, n1, n2 = ctx.rho0.shape
    i0 = jnp.arange(n0)
    i1 = jnp.arange(n1)
    q0 = jnp.where(i0 <= n0 // 2, i0, i0 - n0)
    q1 = jnp.where(i1 <= n1 // 2, i1, i1 - n1)
    q2 = jnp.arange(n2 // 2 + 1)
    qsq = q0[:, None, None] ** 2 + q1[None, :, None] ** 2 + q2[None, None, :] ** 2
    inside = qsq <= q_sigma ** 2
    # Var(L_sigma u) = sum(mult * S_sigma) / N
    level = c * jnp.sqrt((n0 * n1 * n2) / jnp.sum(ctx.mult * inside))
    return jnp.where(inside, level, 0.0)


@jax.jit
def sigma_field(u, ctx):
    """Build the positive amplitude field that modulates the refined density.

    Exponential of a band-limited field, floored at ``s_min``, so it is positive
    everywhere by construction and needs no constraint. At ``u = 0`` it is exactly
    ``s_min + 1``, and with a zero spectrum it is that constant everywhere.
    """
    return ctx.s_min + jnp.exp(apply_mult(u, ctx.sqrtS_sigma))


@jax.jit
def delta_rho(state, ctx):
    """Build the density correction from the two latent fields.

    A band-limited field from ``z``, modulated pointwise by sigma, then projected back
    onto the band-limited symmetric subspace. The projection is the last operation, so
    the result cannot carry power outside ``mask_obs`` -- the quantity the null gate's
    second number measures. The pointwise product is the only place the two fields
    couple, and with constant sigma it cannot move a coefficient between masks.
    """
    return pi_real(sigma_field(state.u, ctx) * apply_mult(state.z, ctx.sqrtS), ctx)


@jax.jit
def rho_total(state, ctx):
    """Add the correction to the starting model.

    In zero-field mode ``rho0`` is exactly zero, so this is the correction alone and
    the refinement builds density from the amplitudes with no model at all.
    """
    return ctx.rho0 + delta_rho(state, ctx)


@jax.jit
def sf(rho, ctx):
    """Compute the structure factors the data term sees.

    The spectrum restricted to measured coefficients. Everything outside ``mask_obs``
    is exactly zero, which is what makes the gradient guards in ``e_work`` and
    ``resid_coeff`` necessary rather than defensive.
    """
    return project_band(r2c(rho), ctx.mask_obs)


@jax.jit
def fit_scale(F, ctx, mask):
    """Fit the overall scale between model and observed amplitudes over one mask.

    Weighted least squares in the amplitudes, with the same ``mult * wgt`` weights the
    data term uses. Because the weights match, the derivative of ``e_work`` with
    respect to k vanishes at this k, so holding it fixed loses nothing -- which is why
    the result is wrapped in ``stop_gradient``: the analytic gradient holds k at its
    fitted value, and differentiating through the fit would compute a different
    quantity.
    """
    a = jnp.abs(jax.lax.stop_gradient(F))
    w = ctx.mult * ctx.wgt * mask
    return jax.lax.stop_gradient(jnp.sum(w * a * ctx.amp) / jnp.sum(w * a ** 2))


@jax.jit
def e_work(F, ctx, k):
    """Weighted sum of squared amplitude residuals over the work set.

    The magnitude is taken through the double-``where`` idiom so that reverse-mode
    differentiation never evaluates ``d|F|/dF`` at ``F = 0``, where it is undefined and
    would produce NaN. Free reflections are not in the sum, which is the whole basis of
    the null argument.
    """
    m = ctx.mask_work & (F != 0)
    mag = jnp.where(m, jnp.abs(jnp.where(m, F, 1.0)), 0.0)
    r = jnp.where(ctx.mask_work, k * mag - ctx.amp, 0.0)
    return jnp.sum(ctx.mult * ctx.wgt * r ** 2)


@jax.jit
def neg_part(rho, ctx):
    """Give the shortfall of the density below its floor.

    Zero wherever the density is at or above the floor, and the signed deficit below
    it. This is the only place the positivity prior reads the map.
    """
    return jnp.minimum(rho - ctx.rho_floor, 0.0)


@jax.jit
def e_pos(rho, ctx):
    """Penalise density below the floor.

    Half the squared shortfall, so the term is smooth with a continuous derivative at
    the floor. It is a real-space prior and touches no reflection, yet with ``mu > 0``
    it reaches free coefficients through the projection -- which is exactly why the
    null gate switches it off.
    """
    return 0.5 * jnp.sum(neg_part(rho, ctx) ** 2)


@jax.jit
def objective(state, ctx):
    """Evaluate the objective and its terms.

    Data term on the work set, positivity weighted by ``mu``, and a unit-variance
    Gaussian prior on each latent field. The scale is refitted here at every
    evaluation, on the work set only, and held fixed for differentiation.

    Returns the value and a ``Terms`` carrying each contribution and the fitted scale,
    so a caller can see which term moved without recomputing anything.
    """
    rho = rho_total(state, ctx)
    F = sf(rho, ctx)
    k = fit_scale(F, ctx, ctx.mask_work)
    ew = e_work(F, ctx, k)
    ep = e_pos(rho, ctx)
    ez = 0.5 * jnp.sum(state.z ** 2)
    eu = 0.5 * jnp.sum(state.u ** 2)
    return ew + ctx.mu * ep + ez + eu, Terms(ew, ep, ez, eu, k)


@jax.jit
def resid_coeff(F, ctx, k):
    """Derivative of the data term with respect to the spectrum, on the work set.

    Carries ``wgt`` but **not** ``mult``: the multiplicity already lives in the adjoint,
    since ``c2r`` is the adjoint of ``r2c`` under the multiplicity-weighted inner
    product. Applying it here as well would double-count it on every coefficient off
    the self-conjugate planes. Guarded by the same double-``where`` as ``e_work``.
    """
    m = ctx.mask_work & (F != 0)
    safe = jnp.where(m, F, 1.0)
    mag = jnp.abs(safe)
    # mult already carried by the c2r adjoint
    return jnp.where(m, 2.0 * k * ctx.wgt * (k * mag - ctx.amp) * safe / mag, 0.0)


@jax.jit
def grad_rho(state, ctx):
    """Gradient of the objective with respect to the density.

    The data part enters through ``c2r`` of the residual coefficients, which is where
    the Friedel multiplicity is accounted for; the positivity part is already a real
    map. With ``mu = 0`` only the first survives, and it is supported on the work set
    -- the statement the null rests on.
    """
    rho = rho_total(state, ctx)
    F = sf(rho, ctx)
    k = fit_scale(F, ctx, ctx.mask_work)
    return c2r(resid_coeff(F, ctx, k), rho.shape) + ctx.mu * neg_part(rho, ctx)


@jax.jit
def grad(state, ctx):
    """Analytic gradient of the objective with respect to z and u.

    Chain rule through ``delta_rho = pi_real(sigma * L_S z)``. Because ``pi_real`` is
    self-adjoint, the real-space gradient is pushed back through the same projector
    rather than a transpose of it; the ``+ z`` and ``+ u`` terms are the unit-variance
    prior. With ``c = 0`` the sigma spectrum is zero, so ``g_u`` reduces to ``u`` and
    the data can reach z only through coefficients on the work set -- the statement the
    null gate measures.
    """
    v = apply_mult(state.z, ctx.sqrtS)
    sigma = sigma_field(state.u, ctx)
    a = pi_real(grad_rho(state, ctx), ctx)
    g_z = apply_mult(sigma * a, ctx.sqrtS) + state.z
    g_u = apply_mult((sigma - ctx.s_min) * v * a, ctx.sqrtS_sigma) + state.u
    return g_z, g_u


@partial(jax.jit, donate_argnums=(0,))
def step(state, ctx, lr):
    """Take one gradient descent step.

    The state buffers are donated, so the update runs in place and a trajectory never
    accumulates. Callers that need the old state must copy it first.
    """
    g_z, g_u = grad(state, ctx)
    return State(state.z - lr * g_z, state.u - lr * g_u)


@partial(jax.jit, static_argnums=(3,))
def run(state, ctx, lr, n_steps):
    """Run the refinement and emit one trace row per step.

    A ``lax.scan`` over the donated step, so only the carried state and four scalars
    per step exist -- the intermediate maps are never materialised, which is what keeps
    memory independent of the step count.

    ``r_free`` is computed for the trace and for nothing else. It never enters the
    step, the step size, or any stopping decision.
    """
    def body(s, _):
        s = step(s, ctx, lr)
        J, terms = objective(s, ctx)
        F = sf(rho_total(s, ctx), ctx)
        r_work = r_factor(F, ctx, terms.k, ctx.mask_work)
        r_free = r_factor(F, ctx, terms.k, ctx.mask_free)
        return s, Trace(J, terms.e_work, r_work, r_free)

    return jax.lax.scan(body, state, None, length=n_steps)


@jax.jit
def r_factor(F, ctx, k, mask):
    """Compute the crystallographic R factor over one mask.

    Multiplicity-weighted, so a coefficient standing for a Friedel pair counts twice,
    exactly as it would if both members were stored. Invariant to scaling ``amp`` and
    ``k`` together, since both numerator and denominator scale with the amplitudes.
    """
    w = ctx.mult * mask
    return jnp.sum(w * jnp.abs(k * jnp.abs(F) - ctx.amp)) / jnp.sum(w * ctx.amp)


@jax.jit
def delta_r_free(state, ctx):
    """Change in R_free between the starting model and the refined one.

    Both R factors use **one** scale, fitted on the work set at the current state. That
    matters: refitting k per state would move R_free through the scale alone, reporting
    a non-zero result in a configuration where nothing coupled. Holding k fixed across
    the two evaluations isolates the free amplitudes, which is the only quantity this
    project claims to measure.
    """
    F = sf(rho_total(state, ctx), ctx)
    # one work-fitted scale for both R_free values
    k = fit_scale(F, ctx, ctx.mask_work)
    return r_factor(ctx.F0, ctx, k, ctx.mask_free) - r_factor(F, ctx, k, ctx.mask_free)


@jax.jit
def eta_c(state, ctx):
    """Ratio of the change in R_free to the change in R_work.

    How much of the work-set improvement leaked into the test set. Zero when nothing
    coupled, and it uses the same single work-fitted scale for all four R factors, for
    the same reason ``delta_r_free`` does.
    """
    F = sf(rho_total(state, ctx), ctx)
    k = fit_scale(F, ctx, ctx.mask_work)
    dr_work = r_factor(ctx.F0, ctx, k, ctx.mask_work) - r_factor(F, ctx, k, ctx.mask_work)
    return delta_r_free(state, ctx) / dr_work


def knob_grid(index, spec):
    """Map a flat index to one point of the knob grid.

    C order, so ``c`` is the slowest axis and ``mu`` the fastest. Pure and total: a
    SLURM array index addresses a grid point directly, and an out-of-range index
    raises rather than wrapping.
    """
    cs, qs, mus = spec
    if not 0 <= index < len(cs) * len(qs) * len(mus):
        raise IndexError(f"knob index {index} outside grid of {len(cs) * len(qs) * len(mus)}")
    i, rest = divmod(index, len(qs) * len(mus))
    j, l = divmod(rest, len(mus))
    return Knobs(cs[i], qs[j], mus[l])


def identity_sym(h_shape):
    """Build symmetry arrays for P1: one operator, no phase, no conjugation.

    The gather is the identity permutation and ``sym_conj`` is all False, so
    ``project_sym`` returns its argument unchanged.
    """
    idx = jnp.arange(int(np.prod(h_shape)), dtype=jnp.int32).reshape((1,) + tuple(h_shape))
    return (
        idx,
        jnp.ones((1,) + tuple(h_shape), dtype=jnp.complex128),
        jnp.zeros((1,) + tuple(h_shape), dtype=bool),
    )


def make_ctx(key, shape, **overrides):
    """Build a synthetic context from an explicit key, for tests.

    Generates a ground truth map and sets ``amp`` on ``mask_obs`` to its amplitudes, so
    a perfect fit is attainable and the null has something real to be measured against.
    Masks are made Friedel-symmetric and the work/free split disjoint, since both are
    preconditions of the null argument rather than conveniences. Asserts the mask
    identity before returning.
    """
    n0, n1, n2 = shape
    h = (n0, n1, n2 // 2 + 1)
    k_true, k_err, k_free, k_wgt = jax.random.split(key, 4)
    i0 = np.arange(n0)
    i1 = np.arange(n1)
    q0 = np.where(i0 <= n0 // 2, i0, i0 - n0)
    q1 = np.where(i1 <= n1 // 2, i1, i1 - n1)
    q2 = np.arange(h[2])
    qsq = q0[:, None, None] ** 2 + q1[None, :, None] ** 2 + q2[None, None, :] ** 2
    planes = [0] + ([n2 // 2] if n2 % 2 == 0 else [])
    mult = np.full(h, 2, dtype=np.int32)
    mult[:, :, planes] = 1
    flat = np.arange(np.prod(h)).reshape(h)
    mate = flat.copy()
    # Friedel mate within self-conjugate planes
    mate[:, :, planes] = flat[np.ix_((-i0) % n0, (-i1) % n1, planes)]
    mate = mate.ravel()

    env = np.exp(-qsq / (2.0 * (0.25 * min(shape)) ** 2))
    obs = qsq <= (0.45 * min(shape)) ** 2
    rho_true = c2r(r2c(jax.random.normal(k_true, shape, dtype=jnp.float64)) * env, shape)
    err = c2r(r2c(jax.random.normal(k_err, shape, dtype=jnp.float64)) * env, shape)
    rho0 = rho_true + 0.5 * err

    a_true = np.abs(np.asarray(r2c(rho_true)))
    a_true = 0.5 * (a_true + a_true.ravel()[mate].reshape(h))
    draw = np.asarray(jax.random.uniform(k_free, h, dtype=jnp.float64))
    draw = np.minimum(draw, draw.ravel()[mate].reshape(h))
    free = obs & (draw < 0.15)
    free[0, 0, 0] = False
    work = obs & ~free
    wgt = 0.5 + np.asarray(jax.random.uniform(k_wgt, h, dtype=jnp.float64))
    wgt = 0.5 * (wgt + wgt.ravel()[mate].reshape(h))
    sym_idx, sym_phase, sym_conj = identity_sym(h)

    ctx = Ctx(
        rho0=rho0, F0=r2c(rho0), amp=jnp.asarray(np.where(obs, a_true, 0.0)),
        wgt=jnp.asarray(wgt), mult=jnp.asarray(mult),
        mask_work=jnp.asarray(work), mask_free=jnp.asarray(free), mask_obs=jnp.asarray(obs),
        sqrtS=jnp.asarray(env), sqrtS_sigma=jnp.zeros(h, dtype=jnp.float64),
        sym_idx=sym_idx, sym_phase=sym_phase, sym_conj=sym_conj,
        s_min=jnp.asarray(0.1), rho_floor=jnp.asarray(0.0), mu=jnp.asarray(0.0),
    )
    ctx = ctx._replace(sqrtS_sigma=sigma_spectrum(0.3, 2.0, ctx))
    ctx = ctx._replace(**overrides)
    assert bool(jnp.all(ctx.mask_obs == (ctx.mask_work | ctx.mask_free))), (
        "mask_obs must equal mask_work | mask_free"
    )
    return ctx
