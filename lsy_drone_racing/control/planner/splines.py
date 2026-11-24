"""JAX implementation of splines."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spline:
    """Piecewise polynomial in the Bernstein basis.

    Segment k covers [knots[k], knots[k+1]] and is evaluated in the local coordinate
    t = (x - knots[k]) / h_k in [0, 1].

    Evaluation is split into two parts:
        _locate(x)         -> (k, t)   depends on the knot structure
        _eval_local(c, t)  -> point    pure local Bernstein
    """

    coeffs: Array  # (n_seg, degree+1, dim)
    knots: Array  # (n_seg+1,)
    degree: int = field(metadata=dict(static=True))
    uniform: bool = field(default=False, metadata=dict(static=True))

    @property
    def n_seg(self) -> int:
        """Number of segments (static, derived from the shape)."""
        return self.coeffs.shape[0]

    def eval(self, x: Array) -> Array:
        """x: any shape (...) -> (..., dim)."""
        return jnp.vectorize(self._eval_scalar, signature="()->(d)")(x)

    def _eval_scalar(self, x: Array) -> Array:
        """Evaluate the spline at a scalar parameter x.

        Returns:
            (dim,) point on the curve.
        """
        k, t = self._locate(x)
        return self._eval_local(self.coeffs[k], t)

    def derivative(self) -> "Spline":
        """Derivative with respect to the global parameter x, as a new spline of degree-1.

        Bernstein derivative: n * (c[j+1] - c[j]) are the coefficients w.r.t. t,
        and the chain rule d/dx = (1/h_k) d/dt adds the per-segment 1/h.
        """
        if self.degree == 0:  # static check, fine under jit
            raise ValueError("Cannot differentiate a degree-0 spline.")
        h = jnp.diff(self.knots)  # (n_seg,)
        dc = self.degree * (self.coeffs[:, 1:] - self.coeffs[:, :-1]) / h[:, None, None]
        return Spline(coeffs=dc, knots=self.knots, degree=self.degree - 1, uniform=self.uniform)

    def _locate(self, x: Array) -> tuple[Array, Array]:
        """Find segment index k and local coordinate t for parameter x."""
        if self.uniform:
            h = self.knots[1] - self.knots[0]
            k = jnp.floor((x - self.knots[0]) / h).astype(jnp.int32)  # O(1)
        else:
            k = jnp.searchsorted(self.knots, x, side="right") - 1  # O(log n)
        k = jnp.clip(k, 0, self.n_seg - 1)
        t = (x - self.knots[k]) / (self.knots[k + 1] - self.knots[k])
        return k, t

    @staticmethod
    def _eval_local(c: Array, t: Array) -> Array:
        """Evaluate one Bernstein polynomial with coefficients c (degree+1, dim) at t.

        De Casteljau: repeated lerps between neighbouring control points. The loop runs
        over the static degree, so it is unrolled at trace time.

        """
        for _ in range(c.shape[0] - 1):
            c = (1.0 - t) * c[:-1] + t * c[1:]
        return c[0]

    @classmethod
    def from_hermite(
        cls, u: Array, p: Array, v: Array, a: Array, uniform: bool = False
    ) -> "Spline":
        """Create a quintic Hermite spline from knot positions, velocities and accelerations.

        Args:
            u: (K,) strictly increasing knots.
            p: (K, dim) positions at the knots.
            v: (K, dim) first derivative w.r.t. u at the knots.
            a: (K, dim) second derivative w.r.t. u at the knots.
            uniform: Set True only if the knots are equally spaced (enables O(1) lookup).
        """
        u = jnp.asarray(u)
        return cls(coeffs=quintic_hermite(u, p, v, a), knots=u, degree=5, uniform=uniform)


def quintic_hermite(u: Array, p: Array, v: Array, a: Array) -> Array:
    """Compute coefficients of quintic hermite spline in the bernstein basis.

    Args:
        u: (K, ) strictly increasing knots
        p: (K, dim) positions at the knots
        v: (K, dim) first derivative w.r.t. u at the knots
        a: (K, dim) second derivative w.r.t. u at the knots

    Returns:
        cofficients of the spline in bernstein basis.
    """

    def segment_interpolation(u: Array, p: Array, v: Array, a: Array) -> Array:
        """Interpolate quintic spline in segment between two points.

        We parameterize in the local frame by t = (x - u[0]) / h in [0, 1]
        With that compute closed form solutions for beta_0, ..., beta_5

        Args:
        u: (2, ) start and end knot
        p: (2, dim) positions at the knots
        v: (2, dim) first derivative w.r.t. u at the knots
        a: (2, dim) second derivative w.r.t. u at the knots

        Returns:
        (6, dim) coefficients in the bernstein basis of a quintic polynomial.
        """
        h = u[1] - u[0]

        beta_0 = p[0]
        beta_1 = p[0] + (h / 5) * v[0]
        beta_2 = p[0] + 2 * (h / 5) * v[0] + (h * h / 20) * a[0]

        beta_5 = p[1]
        beta_4 = p[1] - (h / 5) * v[1]
        beta_3 = p[1] - 2 * (h / 5) * v[1] + (h * h / 20) * a[1]

        return jnp.stack([beta_0, beta_1, beta_2, beta_3, beta_4, beta_5], dtype=p.dtype)

    u_pairs = jnp.stack([u[:-1], u[1:]], axis=1)
    p_pairs = jnp.stack([p[:-1], p[1:]], axis=1)
    v_pairs = jnp.stack([v[:-1], v[1:]], axis=1)
    a_pairs = jnp.stack([a[:-1], a[1:]], axis=1)

    return jax.vmap(segment_interpolation)(u_pairs, p_pairs, v_pairs, a_pairs)
