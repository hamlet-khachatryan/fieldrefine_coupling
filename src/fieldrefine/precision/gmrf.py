import math
from functools import partial
from itertools import product

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import core

ELL = 0.1
FEATURE_SPAN = 12.0
J_MAX = 8
RQ_ALPHA = 2.0

SHIFT_INVARIANT = ("gaussian", "rq")
KERNELS = ("gaussian", "inner", "rq")


def _offsets(w):
    r = w // 2
    return tuple(product(range(-r, r + 1), repeat=3))


@partial(jax.jit, static_argnums=(1, 2))
def patch_features(rho0, w, stride):
    """Gather the local neighbourhood of every voxel as a raw feature vector.

    The gather is by ``jnp.roll``, so the patch wraps at the cell boundary: the unit
    cell is a torus and the voxel at index 0 is adjacent to the one at index n-1. A
    non-periodic gather would make the boundary features depend on where the origin was
    put, which is a property of the indexing rather than of the density.
    """
    cols = [jnp.roll(rho0, (-o0, -o1, -o2), axis=(0, 1, 2))[::stride, ::stride, ::stride]
            for o0, o1, o2 in _offsets(w)]
    return jnp.stack([c.ravel() for c in cols], axis=1)


@partial(jax.jit, static_argnums=(0,))
def position_features(shape):
    """Return the fractional coordinate of every voxel, and nothing else.

    The negative control of S7. One period of each channel is exactly one cell, which is
    what makes the harmonics drawn by ``rf_block`` the reciprocal lattice of this feature
    space and hence the induced graph exactly circulant.
    """
    axes = [jnp.arange(n, dtype=jnp.float64) / n for n in shape]
    grid = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([g.ravel() for g in grid], axis=1)


@partial(jax.jit, static_argnums=(1,))
def structure_tensor(rho0, sigma):
    """Return the six independent components of the smoothed gradient outer product.

    Derivatives and smoothing are both spectral, so no finite-difference stencil
    introduces a direction the density does not have.
    """
    shape = rho0.shape
    qs = [jnp.fft.fftfreq(shape[0]), jnp.fft.fftfreq(shape[1]), jnp.fft.rfftfreq(shape[2])]
    q = [qs[0][:, None, None], qs[1][None, :, None], qs[2][None, None, :]]
    F = core.r2c(rho0)
    smooth = jnp.exp(-2.0 * (jnp.pi * sigma) ** 2 * (q[0] ** 2 + q[1] ** 2 + q[2] ** 2))
    g = [core.c2r(F * (2j * jnp.pi * qa), shape) for qa in q]
    cols = [core.c2r(core.r2c(g[a] * g[b]) * smooth, shape).ravel()
            for a, b in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    return jnp.stack(cols, axis=1)


@partial(jax.jit, static_argnums=(1,))
def solvent_distance(mask, n_steps):
    """Return the toroidal Manhattan distance from every voxel to the solvent.

    A min-plus relaxation over the six face neighbours, wrapped, so the distance is the
    one on the torus rather than on a box. Distances saturate at ``n_steps``.
    """
    d0 = jnp.where(mask, 0.0, jnp.asarray(n_steps, dtype=jnp.float64))

    def body(_, d):
        for axis in (0, 1, 2):
            d = jnp.minimum(d, jnp.minimum(jnp.roll(d, 1, axis), jnp.roll(d, -1, axis)) + 1.0)
        return d

    return jax.lax.fori_loop(0, n_steps, body, d0).ravel()[:, None]


def concat_features(*phis, whiten=True):
    """Join feature blocks and put every channel on the same torus.

    Whitening alone fixes the scale but not the period. The harmonics of ``rf_block``
    have period 1 in every channel, so a whitened channel is additionally divided by
    ``FEATURE_SPAN``: six standard deviations then span half a period and the wrap-round
    coupling between the extremes of a channel falls below the kernel's floor. Positions
    already occupy exactly one period and must not be passed through here.
    """
    phi = jnp.concatenate([jnp.atleast_2d(p) for p in phis], axis=1)
    if not whiten:
        return phi
    mean = jnp.mean(phi, axis=0, keepdims=True)
    sd = jnp.std(phi, axis=0, keepdims=True)
    return (phi - mean) / (jnp.where(sd > 0.0, sd, 1.0) * FEATURE_SPAN)


@jax.jit
def embed_cnn(params, rho0):
    """Map the density to a learned per-voxel embedding.

    One periodic convolution followed by ``tanh``, so the output is bounded and lands on
    the same torus the fixed features use. Phase B only; the freeze rule of S7 still
    applies, so this is evaluated once from ``rho0`` and held.
    """
    w = params["w"]
    patches = patch_features(rho0, w.shape[0], 1)
    return jnp.tanh(patches @ w.reshape(-1, w.shape[-1]) + params["b"]) / FEATURE_SPAN


def _harmonics(seed, m_half, d, kernel):
    k_j, k_s = jax.random.split(jax.random.PRNGKey(seed))
    sigma = jnp.full((m_half, 1), 1.0 / (2.0 * jnp.pi * ELL), dtype=jnp.float64)
    if kernel == "rq":
        # rational quadratic is a gamma mixture of gaussians
        sigma = sigma * jnp.sqrt(jax.random.gamma(k_s, RQ_ALPHA, (m_half, 1)) / RQ_ALPHA)
    j = jnp.arange(-J_MAX, J_MAX + 1, dtype=jnp.float64)
    logits = jnp.broadcast_to(
        (-0.5 * (j / sigma) ** 2)[:, None, :], (m_half, d, 2 * J_MAX + 1)
    )
    return jax.random.categorical(k_j, logits, axis=-1).astype(jnp.float64) - J_MAX


def _root_width(m):
    r = math.isqrt(int(m))
    return max(2, r - (r % 2))


@partial(jax.jit, static_argnums=(1, 2, 3))
def rf_block(phi_chunk, seed, m, kernel):
    """Build the random-feature rows of one chunk of voxels.

    The rows are the tensor square of an underlying feature ``y``: ``Z_i = vec(y_i
    y_i^T)``, so ``(Z Z^T)_ij = (y_i . y_j)^2``. The square is what makes the graph
    usable at all. ``Z Z^T`` being PSD constrains the *matrix*; it says nothing about the
    *entries*, and ``L = D - W`` is PSD only for non-negative weights, since
    ``x'Lx = 0.5 sum w_ij (x_i - x_j)^2`` carries the sign of each w_ij. A Gaussian
    kernel is pointwise non-negative but its finite-m random-feature estimate is not:
    measured over a 27-channel patch embedding, half the entries came out negative and
    ``lambda_min(L)`` reached -75. Squaring an elementwise-signed kernel makes every
    weight non-negative by construction, at any m, and costs only a halving of the
    length scale -- ``k^2`` is the same Gaussian at ``ell/sqrt(2)``.

    Squaring also preserves the two properties the square root already had. Elementwise
    squaring of a circulant matrix is circulant, so a position-only graph stays exactly a
    convolution; and the inner-product kernel stays outside the admissibility theorem, so
    it remains a live positive control.

    The underlying ``y`` uses cosine and sine of the **same** frequency, in pairs, so that
    ``(Y Y^T)_ij`` collapses to ``(2/r) sum_l cos(2 pi j_l . (phi_i - phi_j))``: a function
    of the feature difference alone, exactly, at every r. The random-phase variant
    ``cos(w.phi + b)`` leaves a term in ``phi_i + phi_j`` that vanishes only in
    expectation, and with it the admissibility theorem of S7 would hold only to
    O(r^-1/2). The frequencies are integer harmonics drawn from the periodised spectral
    density: on a torus Bochner's theorem gives a discrete spectral measure, and a lattice
    frequency is what makes a position-only graph exactly circulant rather than merely
    Toeplitz -- the difference between test 11 reading round-off and reading 1e-3.

    ``m`` is the width of Z, so the underlying harmonic count is ``sqrt(m)``; columns past
    ``r^2`` are zero and contribute nothing to ``Z Z^T``.
    """
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}, expected one of {KERNELS}")
    d = phi_chunk.shape[1]
    r = _root_width(m)
    if kernel == "inner":
        omega = jax.random.normal(jax.random.PRNGKey(seed), (d, r), dtype=jnp.float64)
        y = phi_chunk @ omega / jnp.sqrt(r)
    else:
        a = 2.0 * jnp.pi * (phi_chunk @ _harmonics(seed, r // 2, d, kernel).T)
        y = jnp.sqrt(2.0 / r) * jnp.concatenate([jnp.cos(a), jnp.sin(a)], axis=1)
    z = (y[:, :, None] * y[:, None, :]).reshape(phi_chunk.shape[0], r * r)
    return jnp.pad(z, ((0, 0), (0, m - r * r)))


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def apply_zzt(phi, seed, m, kernel, chunk, x):
    """Apply ``Z Z^T`` to a flat field without ever holding Z.

    Two passes over row chunks: the first accumulates ``s = Z^T x``, the second writes
    ``Z s``. One chunk of Z exists at a time, so the footprint is the stored phi plus
    ``chunk * m``, not ``N * m``. Rows past the end of the grid are padded and masked to
    zero, since a padded row of zeros would otherwise contribute ``cos(0) = 1`` to every
    feature and add a phantom voxel to the graph.

    ``degree`` and the matvec both go through here, so ``d * 1`` and ``Z(Z^T 1)`` are the
    same floating-point expression and cancel bitwise.
    """
    n, d = phi.shape
    pad = (-n) % chunk
    nb = (n + pad) // chunk
    phi_p = jnp.pad(phi, ((0, pad), (0, 0))).reshape(nb, chunk, d)
    x_p = jnp.pad(x, (0, pad)).reshape(nb, chunk)
    keep = jnp.pad(jnp.ones(n, dtype=jnp.float64), (0, pad)).reshape(nb, chunk)

    def forward(acc, arg):
        p, xx, kk = arg
        return acc + (rf_block(p, seed, m, kernel) * kk[:, None]).T @ xx, None

    s, _ = jax.lax.scan(forward, jnp.zeros(m, dtype=jnp.float64), (phi_p, x_p, keep))

    def backward(carry, arg):
        p, kk = arg
        return carry, (rf_block(p, seed, m, kernel) * kk[:, None]) @ s

    _, y = jax.lax.scan(backward, None, (phi_p, keep))
    return y.reshape(-1)[:n]


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def degree(phi, seed, m, kernel, chunk):
    """Return the row sums of W, by the same two-pass structure as the matvec."""
    return apply_zzt(phi, seed, m, kernel, chunk, jnp.ones(phi.shape[0], dtype=jnp.float64))


@partial(jax.jit, static_argnums=(1, 2))
def knn_graph(phi, k, chunk=1024):
    """Return the k nearest neighbours of every voxel in feature space, and their weights.

    The sparse alternative to the low-rank form. Weights use the same length scale as the
    random features, so the two paths describe the same kernel and test 5 compares
    approximations rather than two different graphs. The self edge is excluded; the
    relation is not symmetric, and the caller symmetrises before forming D - W.

    Distances are computed in row chunks, so the footprint is ``chunk * N`` rather than
    ``N^2`` -- at N = 1e6 the square would be 8 TB. The **time** is still O(N^2 d): this
    is exact brute force, tractable to roughly 1e5 voxels. Beyond that the sparse path
    needs a spatial index, which would mean a dependency S1 does not allow.
    """
    n = phi.shape[0]
    pad = (-n) % chunk
    sq = jnp.sum(phi ** 2, axis=1)
    rows = jnp.pad(phi, ((0, pad), (0, 0))).reshape(-1, chunk, phi.shape[1])
    offs = jnp.arange(n + pad).reshape(-1, chunk)

    def block(_, arg):
        p, off = arg
        d2 = jnp.sum(p ** 2, axis=1)[:, None] + sq[None, :] - 2.0 * (p @ phi.T)
        d2 = jnp.where(off[:, None] == jnp.arange(n)[None, :], jnp.inf, d2)
        i = jnp.argsort(d2, axis=1)[:, :k]
        return None, (i, jnp.take_along_axis(d2, i, axis=1))

    _, (idx, near) = jax.lax.scan(block, None, (rows, offs))
    idx = idx.reshape(-1, k)[:n]
    near = near.reshape(-1, k)[:n]
    # the low-rank path squares its kernel; match it, or test 5 compares two graphs
    return idx.astype(jnp.int32), jnp.exp(-jnp.maximum(near, 0.0) / ELL ** 2)


def graph_spectrum(apply_L, n_eig, key, shape):
    """Return the Ritz values of a matrix-free operator by Lanczos.

    Full reorthogonalisation against the stored basis: without it the Lanczos vectors
    lose orthogonality after a few steps and the Ritz values acquire spurious copies of
    the dominant eigenvalue, which would be read as a condition number rather than as a
    breakdown.
    """
    v = jax.random.normal(key, shape, dtype=jnp.float64)
    v = v / jnp.linalg.norm(v.ravel())
    basis, alphas, betas = [], [], []
    beta, prev = 0.0, jnp.zeros(shape, dtype=jnp.float64)
    for i in range(n_eig):
        basis.append(v)
        w = apply_L(v)
        alpha = float(jnp.sum(w * v))
        w = w - alpha * v - beta * prev
        for b in basis:
            w = w - jnp.sum(w * b) * b
        beta = float(jnp.linalg.norm(w.ravel()))
        alphas.append(alpha)
        if beta <= 1e-12 * max(abs(alpha), 1.0) or i == n_eig - 1:
            break
        betas.append(beta)
        prev, v = v, w / beta
    a = np.asarray(alphas)
    b = np.asarray(betas)
    return jnp.asarray(np.linalg.eigvalsh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1)))


def effective_rank(evals, tau):
    """Count the eigenvalues carrying more than a fraction tau of the largest."""
    e = jnp.asarray(evals)
    return int(jnp.sum(e > tau * jnp.max(e)))
