"""Reference path of the high level planner, parameterized by its progress (arc length) s.

The path is a quintic Hermite spline through labelled knots: the start, the gates, the final
waypoint (all fixed) and extra waypoints that the planner inserts or replaces, e.g. to push the
path away from obstacles. Knots are parameterized by chord length first, then the spline is
resampled densely and refitted in arc length, so that s measures the distance along the path.

References are immutable: every update returns a new reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BPoly

if TYPE_CHECKING:
    from numpy.typing import NDArray


def quintic_hermite(u: NDArray, p: NDArray, v: NDArray, a: NDArray) -> BPoly:
    """Create a quintic hermite spline parameterized in chord lengths u."""
    if np.any(np.diff(u) <= 0):
        raise ValueError("Consecutive knots must not coincide.")

    curve = BPoly.from_derivatives(u, np.stack([p, v, a], axis=1))  # (K, 3 derivs, 3 dims)
    return curve


def arclen_derivs(dp_du: NDArray, d2p_du2: NDArray) -> tuple[NDArray, NDArray]:
    """Apply chain rule to get arclength derivatives from derivatives w.r.t. chord length u."""
    ds_du = la.norm(dp_du, axis=-1, keepdims=True)
    tangent = dp_du / ds_du  # dp_ds

    d2p_du_tan = np.sum(d2p_du2 * tangent, axis=-1, keepdims=True) * tangent
    curvature = (d2p_du2 - d2p_du_tan) / ds_du**2
    return tangent, curvature


class ProgressReference:
    """Progress-based reference path."""

    START_END_LABEL: int = -1
    GATES_LABEL: int = 0
    UPDATEABLE_LABEL: int = 1

    def __init__(
        self,
        p: NDArray,
        v: NDArray,
        a: NDArray,
        pt_labels: NDArray,
        tangent_scale: NDArray,
        M: int = 32,
    ):
        """Initialize a progress-based reference path.

        Args:
            p, v, a:  positions, tangents, accelerations of all slots.
            tangent_scale
            pt_labels: 0: fixed, 1: extra
            M:                    table samples per segment.
        """
        self._p, self._v, self._a = p, v, a
        self._tangent_scale = tangent_scale
        self._pt_labels = pt_labels
        self._M = M

        self._build()

    def _build(self):
        p, v, a = self._p, self._v, self._a

        self._u = np.append(0, np.cumsum(la.norm(np.diff(p, axis=0), axis=1)))  # (K, )

        # sparse
        self._spline_u = quintic_hermite(self._u, p, v, a)

        # Reparameterization by arc length of the generated spline
        u_per_segment = np.linspace(
            self._u[:-1], self._u[1:], self._M, axis=-1, endpoint=False
        )  # (K-1, M)
        self._u_samples = np.append(u_per_segment.ravel(), self._u[-1])  # ((K - 1) * M + 1, )
        # compute progress values numerically
        dp_du = self._spline_u(self._u_samples, nu=1)  # p' ((K - 1) * M + 1, 3)
        speed = la.norm(dp_du, axis=-1)  # ||p'|| ((K - 1) * M + 1, )
        # integrate: s(u) = \integral ||p'|| du
        self._s_samples = cumulative_trapezoid(
            speed, self._u_samples, initial=0.0
        )  # ((K - 1) * M + 1, )

        # parameterize second spline in progress
        p = self._spline_u(self._u_samples, nu=0)
        v_s, a_s = arclen_derivs(
            self._spline_u(self._u_samples, nu=1), self._spline_u(self._u_samples, nu=2)
        )
        # dense
        self._spline_s = quintic_hermite(self._s_samples, p, v_s, a_s)

        self.length = self._s_samples[-1]

    @property
    def dense_s(self) -> NDArray:
        """Dense samples in s."""
        return self._s_samples

    @classmethod
    def initial_reference(
        cls,
        ego_pos: NDArray,
        gates_pos_seq: NDArray,
        gates_normals: NDArray,
        tangent_scale: float | NDArray = 1.0,
    ) -> "ProgressReference":
        """Build an initial reference.

        Args:
        ego_pos: (1, 3, )
        gates_pos_seq: (n_gates, 3)
        gates_normals: (n_gates, 3)
        tangent_scale: (n_gates + 1, ) | scalar scales the gates normals
        """
        n_gates = len(gates_pos_seq)  # number of gates in the sequence
        tangent_scale = np.broadcast_to(tangent_scale, (n_gates + 2,))  # +1 for ego

        p = np.zeros((n_gates + 2, 3))
        v = np.zeros_like(p)
        a = np.zeros_like(p)

        pt_lables = np.zeros((n_gates + 2,))
        pt_lables[0] = cls.START_END_LABEL
        pt_lables[1:-1] = cls.GATES_LABEL
        pt_lables[-1] = cls.START_END_LABEL

        ego_normal = gates_pos_seq[0] - ego_pos
        ego_normal = ego_normal / la.norm(ego_normal)

        # initial position
        p[0] = ego_pos
        v[0] = tangent_scale[0, None] * ego_normal

        # gates positions
        p[1:-1] = gates_pos_seq
        v[1:-1] = tangent_scale[1:-1, None] * gates_normals

        # final waypoint: extend along last gates normal
        p[-1] = gates_pos_seq[-1] + gates_normals[-1]
        v[-1] = tangent_scale[-1, None] * gates_normals[-1]

        return cls(p, v, a, pt_lables, tangent_scale[1:-1])

    def __call__(self, s: NDArray, nu: int = 0) -> NDArray:
        """Evaluate the path at progress s.

        Beyond the end of the path, it continues straight along the final tangent.

        Args:
            s: Progress value(s) [m], scalar or array.
            nu: 0 for the position, 1 for the unit tangent, 2 for the curvature vector.

        Returns:
            Array of shape (..., 3).
        """
        s = np.asarray(s)
        s_in = np.clip(s, 0.0, self.length)
        out = self._spline_s(s_in, nu=nu)

        # beyond > 0 at the end of the path.
        beyond = (s - s_in)[..., None]
        if nu == 0:
            return out + beyond * self._spline_s(self.length, nu=1)  # straight continuation
        if nu == 2:
            return np.where(beyond > 0.0, 0.0, out)  # no curvature beyond end

        return out

    def project(
        self,
        pos: NDArray,
        s_pred: NDArray,
        behind: float = 0.1,
        ahead: float = 0.3,
        n_cand: int = 32,
        gn_steps: int = 5,
    ) -> NDArray:
        """Find progress of closest point to pos on spline."""
        s_cand = np.clip(s_pred + np.linspace(-behind, ahead, n_cand), 0, self.length)
        pts = self(s_cand)  # (n_cand, 3)

        i = np.argmin(la.norm(pts - pos[None, :], axis=-1))
        s0 = s_cand[i]

        ds = (behind + ahead) / (n_cand - 1)
        s = s0
        # max. gn_steps (clipped) gauss newton steps
        for _ in range(gn_steps):
            step = (pos - self(s)) @ self(s, 1)
            if abs(step) < 1e-7:
                break
            s = np.clip(s + step, s0 - ds, s0 + ds)

        s = np.clip(s, 0, self.length)

        return s

    def update_gates(
        self, gates_pos: NDArray, gates_normals: NDArray, mask: NDArray
    ) -> "ProgressReference":
        """Create a ProgressReference with changed gate positions.

        We provide all of the variables with fixed shapes, to avoid triggering recompiles
        Args:
            gates_pos
            gates_normals
            mask
        """
        p, v, a = self._p.copy(), self._v.copy(), self._a.copy()

        slots = self._get_gate_slots()[mask]
        p[slots] = gates_pos[mask]
        v[slots] = self._tangent_scale[mask, None] * gates_normals[mask]
        a[slots] = 0.0

        return ProgressReference(
            p, v, a, pt_labels=self._pt_labels, tangent_scale=self._tangent_scale, M=self._M
        )

    def without_extras(self) -> "ProgressReference":
        """Set mask invalid for all current extra waypoints."""
        p, v, a = self._p.copy(), self._v.copy(), self._a.copy()

        keep = self._pt_labels != self.UPDATEABLE_LABEL

        return ProgressReference(
            p[keep],
            v[keep],
            a[keep],
            pt_labels=self._pt_labels[keep],
            tangent_scale=self._tangent_scale,
            M=self._M,
        )

    def update_final_waypoint(
        self, final_p: NDArray, final_v: NDArray, final_a: NDArray
    ) -> "ProgressReference":
        """Update the final waypoint of the reference."""
        p, v, a = self._p.copy(), self._v.copy(), self._a.copy()

        p[-1] = final_p
        v[-1] = final_v
        a[-1] = final_a

        return ProgressReference(
            p, v, a, pt_labels=self._pt_labels, tangent_scale=self._tangent_scale, M=self._M
        )

    def update_extras(
        self,
        s: NDArray,
        p: NDArray,
        v: NDArray,
        a: NDArray,
        min_spacing: float = 0.3,
        min_fixed_dist: float = 0.1,
    ) -> "ProgressReference":
        """Insert extra waypoints, or replace existing extras closer than min_spacing.

        Start, end and gate knots are never replaced: new points closer than min_fixed_dist
        (in s) to one of them are dropped instead.

        Args:
            s: (n_points,) progress on the current reference where the points belong.
            p: (n_points, 3) positions.
            v: (n_points, 3) first derivatives.
            a: (n_points, 3) second derivatives.
            min_spacing: Existing extras closer than this (in s) to a new point are replaced.
            min_fixed_dist: New points closer than this (in s) to a fixed knot are dropped.

        Returns:
            New reference with all knots ordered by their progress s.
        """
        s = np.asarray(s)
        s_knots = self._s_samples[:: self._M]  # (K,) progress of every knot
        is_extra = self._pt_labels == self.UPDATEABLE_LABEL

        # fixed knots cannot give way, so new points too close to them are dropped
        too_close = (np.abs(s[:, None] - s_knots[~is_extra][None, :]) < min_fixed_dist).any(1)
        s, p, v, a = s[~too_close], p[~too_close], v[~too_close], a[~too_close]
        if s.size == 0:
            return self

        # old extras within min_spacing of a new point are replaced by it
        near = (np.abs(s_knots[:, None] - s[None, :]) < min_spacing).any(axis=1)  # (K,)
        keep = ~(is_extra & near)

        S = np.concatenate([s_knots[keep], s])
        P = np.concatenate([self._p[keep], p])
        V = np.concatenate([self._v[keep], v])
        A = np.concatenate([self._a[keep], a])
        L = np.concatenate([self._pt_labels[keep], np.full(len(s), self.UPDATEABLE_LABEL)])

        order = np.argsort(S, kind="stable")

        return ProgressReference(
            P[order],
            V[order],
            A[order],
            pt_labels=L[order],
            tangent_scale=self._tangent_scale,
            M=self._M,
        )

    def _get_gate_slots(self) -> NDArray:
        return np.flatnonzero(self._pt_labels == self.GATES_LABEL)

    def get_current_gate_positions(self, mask: NDArray) -> tuple:
        """Returns current positions of the gates in the reference."""
        slots = self._get_gate_slots()[mask]
        return self._p[slots]

    def sample(self, num_samples: int) -> tuple:
        """Draw num_samples points from spline equally spaced across u."""
        u = np.linspace(0, self._u[-1], num_samples)
        return u, self._spline_u(u)
