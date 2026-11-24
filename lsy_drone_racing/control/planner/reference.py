"""Reference"""

from __future__ import annotations

import warnings
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


# LAYOUT
def num_slots(n_gates: int, n_extra: int) -> int:
    """Number of slots for waypoints in reference."""
    return 2 + n_gates + n_gates * n_extra


def gate_slot(gate_idx: int | NDArray, n_gates: int, n_extra: int) -> int | NDArray:
    """Slot index of the i-th gate in the sequnce."""
    return np.clip(1 + n_extra + (n_extra + 1) * gate_idx, 0, num_slots(n_gates, n_extra) - 1)


def arclen_derivs(dp_du: NDArray, d2p_du2: NDArray) -> tuple[NDArray, NDArray]:
    """Apply chain rule to get arclength derivatives from derivatives w.r.t. chord length u."""
    ds_du = la.norm(dp_du, axis=-1, keepdims=True)
    tangent = dp_du / ds_du  # dp_ds

    d2p_du_tan = np.sum(d2p_du2 * tangent, axis=-1, keepdims=True) * tangent
    curvature = (d2p_du2 - d2p_du_tan) / ds_du**2
    return tangent, curvature


class ProgressReference:
    """Progress-based reference path."""

    def __init__(
        self,
        p: NDArray,
        v: NDArray,
        a: NDArray,
        mask: NDArray,
        tangent_scale: NDArray,
        n_gates: int,
        n_extra: int,
        M: int = 32,
    ):
        """Initialize a progress-based reference path.

        Args:
            p, v, a: (n_slots, 3) positions, tangents, accelerations of all slots
            mask:    (n_slots, )  True = slot is active
            n_gates, n_extra:     gates and extra slots per gap
            M:                    table samples per segment
        """
        self._p, self._v, self._a, self._mask = p, v, a, np.asarray(mask, dtype=bool)
        self._tangent_scale = tangent_scale
        self._n_gates, self._n_extra, self._M = n_gates, n_extra, M

        self._build()

    def _build(self):
        # compact
        p, v, a = self._p[self._mask], self._v[self._mask], self._a[self._mask]

        self._u = np.append(0, np.cumsum(la.norm(np.diff(p, axis=0), axis=1)))  # (K, )

        # sparse
        self._spline_u = quintic_hermite(self._u, p, v, a)

        # Reparameterization by arc length of the generated spline
        # TODO ATTENTION: depends on the amount of points inserted (triggers recompile in jax)
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

    def _gate_slot(self, gate_idx: int | NDArray) -> int | NDArray:
        """Slot index of the i-th gate in the sequnce."""
        return gate_slot(gate_idx, self._n_gates, self._n_extra)

    def _progress_at_slot_idx(self, slot_idx: int | NDArray) -> int | NDArray:
        """Returns the progress at the gate for a given gate_idx."""
        slot_idx = np.asarray(slot_idx)

        knot_idx = np.cumsum(self._mask)[slot_idx] - 1
        return self._s_samples[knot_idx * self._M]

    @property
    def num_slots(self) -> int:
        """Number of slots for waypoints in reference."""
        return num_slots(self._n_gates, self._n_extra)

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
        n_extra: int = 8,
    ) -> "ProgressReference":
        """Build an initial reference.

        Args:
        ego_pos: (1, 3, )
        gates_pos_seq: (n_gates, 3)
        gates_normals: (n_gates, 3)
        tangent_scale: (n_gates + 1, ) | scalar scales the gates normals
        n_extra: number of slots between gates for adjusting the reference online
        """
        n_gates = len(gates_pos_seq)  # number of gates in the sequence
        n_slots = num_slots(n_gates, n_extra)
        tangent_scale = np.broadcast_to(tangent_scale, (n_gates + 2,))  # +1 for ego

        p = np.zeros((n_slots, 3))
        v = np.zeros_like(p)
        a = np.zeros_like(p)

        mask = np.zeros((n_slots,), dtype=np.bool)
        gate_slots = gate_slot(np.arange(n_gates), n_gates, n_extra)

        ego_normal = gates_pos_seq[0] - ego_pos
        ego_normal = ego_normal / la.norm(ego_normal)

        # initial position
        p[0] = ego_pos
        v[0] = tangent_scale[0, None] * ego_normal
        mask[0] = True

        # gates positions
        p[gate_slots] = gates_pos_seq
        v[gate_slots] = tangent_scale[1:-1, None] * gates_normals
        mask[gate_slots] = True

        # final waypoint: extend along last gates normal
        p[-1] = gates_pos_seq[-1] + gates_normals[-1]
        v[-1] = tangent_scale[-1, None] * gates_normals[-1]
        mask[-1] = True

        return cls(p, v, a, mask, tangent_scale[1:-1], n_gates, n_extra)

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

        slots = self._gate_slot(np.flatnonzero(mask))
        p[slots] = gates_pos[mask]
        v[slots] = self._tangent_scale[mask, None] * gates_normals[mask]
        a[slots] = 0.0

        return ProgressReference(
            p,
            v,
            a,
            self._mask,
            tangent_scale=self._tangent_scale,
            n_gates=self._n_gates,
            n_extra=self._n_extra,
            M=self._M,
        )

    def without_extras(self) -> "ProgressReference":
        """Set mask invalid for all current extra waypoints."""
        mask = np.zeros_like(self._mask)
        mask[0] = True
        mask[self._gate_slot(np.arange(self._n_gates))] = True
        mask[-1] = True

        return ProgressReference(
            self._p,
            self._v,
            self._a,
            mask,
            tangent_scale=self._tangent_scale,
            n_gates=self._n_gates,
            n_extra=self._n_extra,
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
            p,
            v,
            a,
            self._mask,
            tangent_scale=self._tangent_scale,
            n_gates=self._n_gates,
            n_extra=self._n_extra,
            M=self._M,
        )

    def update_extras(
        self,
        s: NDArray,
        p: NDArray,
        v: NDArray,
        a: NDArray,
        valid: NDArray | None = None,
        min_spacing: float = 0.3,
    ) -> "ProgressReference":
        """Insert extra waypoints, or replace existing ones closer than min_spacing.

        Args:
            s: (n_points,) progress on the current reference where the points belong.
            p: (n_points, 3) positions.
            v: (n_points, 3) first derivatives.
            a: (n_points, 3) second derivatives.
            valid: (n_points,) bool, only these points are used. Lets callers pass fixed shapes.
            min_spacing: Existing extras closer than this (in s) to a new point are replaced.

        Returns:
            New reference with the extras of every affected gap re-packed in order of s.
        """
        s = np.asarray(s)
        if valid is not None:
            # NumPy-only: the variable-size selection below is what a JAX port replaces with masks
            s, p, v, a = s[valid], p[valid], v[valid], a[valid]
        if s.size == 0:
            return self
        P, V, A, mask = self._p.copy(), self._v.copy(), self._a.copy(), self._mask.copy()

        progress = self._progress_at_slot_idx(np.arange(self.num_slots))  # (num_slots, )
        progress = np.where(self._mask, progress, np.inf)
        gates_progress = progress[self._gate_slot(np.arange(self._n_gates))]
        gaps = np.searchsorted(gates_progress, s, side="right")

        for gap in np.unique(gaps):
            new = np.flatnonzero(gaps == gap)
            if gap >= self._n_gates:
                warnings.warn(f"Dropped {len(new)} extra waypoint(s) after the last gate.")
                continue

            slots = np.arange(self._gate_slot(gap) - self._n_extra, self._gate_slot(gap))
            old = slots[mask[slots]]
            # merge: a new point replaces every old one within min_spacing
            far = np.abs(progress[old][:, None] - s[new][None, :]) >= min_spacing
            old = old[far.all(axis=1)]

            # combine old and new, sort by progress, pack into the gap from the front
            order = np.argsort(np.concatenate([progress[old], s[new]]))
            if len(order) > self._n_extra:
                warnings.warn(f"Gap {gap} is full, dropped {len(order) - self._n_extra} point(s).")
                order = order[: self._n_extra]
            k = len(order)
            P[slots[:k]] = np.concatenate([P[old], p[new]])[order]
            V[slots[:k]] = np.concatenate([V[old], v[new]])[order]
            A[slots[:k]] = np.concatenate([A[old], a[new]])[order]
            mask[slots] = False
            mask[slots[:k]] = True

        return ProgressReference(
            P,
            V,
            A,
            mask,
            tangent_scale=self._tangent_scale,
            n_gates=self._n_gates,
            n_extra=self._n_extra,
            M=self._M,
        )

    def get_current_gate_positions(self, mask: NDArray) -> tuple:
        """Returns current positions of the gates in the reference."""
        slots = self._gate_slot(np.flatnonzero(mask))
        return self._p, slots

    def sample(self, num_samples: int) -> tuple:
        """Draw num_samples points from spline equally spaced across u."""
        u = np.linspace(0, self._u[-1], num_samples)
        return u, self._spline_u(u)
