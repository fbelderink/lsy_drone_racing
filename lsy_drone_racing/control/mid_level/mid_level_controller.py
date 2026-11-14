import numpy as np
import numpy.linalg as la

from lsy_drone_racing.control.planner.reference import Reference


class MidLevelController:
    def __init__(
        self,
        planner,
        v_max,
        a_long_max,
        a_lat_max,
        a_brake_max,
        v_gate_max=None,
        gate_window=0.5,
        gate_plateau=0.0,
        progress_window=(0.2, 0.2),
        thrust_max=None,
        thrust_min=0.0,
        tilt_max=None,
        smooth_time=None,
    ):

        self._planner = planner

        self.v_max = v_max
        # speed cap at a gate's progress, rising linearly back to v_max gate_window [m] progress away
        self.v_gate_max = v_max if v_gate_max is None else v_gate_max
        self.gate_window = gate_window
        # the cap is flat within gate_plateau [m] of a gate's progress, so it holds until the gate is revealed
        self.gate_plateau = gate_plateau
        self.a_long_max = a_long_max
        self.a_lat_max = a_lat_max
        self.a_brake_max = a_brake_max
        # (behind, ahead) [m] window around the predicted progress in which the path is searched
        self.progress_window = progress_window

        # Thrust feasibility of the profile (None disables it). The acceleration demanded by the
        # path, a = v^2 * kappa + a_t * tangent, plus gravity must be a thrust the drone can produce:
        # thrust_min <= |a + g| <= thrust_max [m/s^2] and a tilt of a + g from vertical <= tilt_max.
        # Without it the lateral limit ignores gravity, so crests and descents at speed demand
        # more than 1 g downwards and turns more than the maximum thrust.
        self.thrust_max = thrust_max
        self.thrust_min = thrust_min
        self.tilt_max = tilt_max
        # Width in seconds of a jerk limit (None disables): the profile is replaced by a moving
        # average of its moving minimum over this window in time. This stays below the original
        # profile everywhere and turns every switch between accelerating and braking into a ramp
        # of this duration, which the drone needs to rotate its thrust direction (a switch from
        # full acceleration to full braking turns it by ~90 degrees).
        self.smooth_time = smooth_time

        self._current_reference_version = 0
        self._current_velocity_profile = None

        self._current_progress = 0.0

    def get_trajectory(self, obs, horizon, dt):

        ref = self._planner.plan(obs)
        path = ref.path

        if self._current_reference_version != ref._version:
            self._current_velocity_profile = self._compute_velocity_profile(ref, obs)
            self._current_reference_version = ref._version

        s_k = np.zeros((horizon + 1,))
        v_k = np.zeros((horizon + 1,))

        current_speed = la.norm(obs["vel"])
        # predict progress by integrating the velocity component along the path tangent, then
        # correct it by projecting onto the path in a small window around the prediction
        tangent = path(self._current_progress, 1)
        tangent = tangent / max(la.norm(tangent), 1e-9)
        s_pred = np.clip(self._current_progress + dt * np.dot(obs["vel"], tangent), 0.0, ref.length)
        s0 = ref.get_progress(obs["pos"], prev_progress=s_pred, search_range_s=self.progress_window)

        s_k[0] = s0
        v_k[0] = current_speed

        def velocity_profile(progress):

            i = np.clip(np.searchsorted(ref.path.x, progress), 0, len(ref.path.x) - 1)

            return self._current_velocity_profile[i]

        for k in range(horizon):
            v_k[k + 1] = min(velocity_profile(s_k[k]), v_k[k] + self.a_long_max * dt)
            s_k[k + 1] = s_k[k] + v_k[k] * dt

        ref_window = np.hstack([path(s_k), path(s_k, 1) * v_k[:, None]])

        self._current_progress = s0
        return ref_window  # (H + 1, 6)

    def _compute_velocity_profile(self, ref: Reference, obs, eps=1e-6, pass_tol=0.2):

        # our path is parameterized by arc length
        s = ref.path.x
        # therefore the second derivative is curvature kappa
        kappa = la.norm(ref.path(s, 2), axis=-1)

        # kappa: 1 / R
        # a_lat = v^2 / R
        # => v = sqrt(a_lat / kappa)
        v = np.minimum(self.v_max, np.sqrt(np.maximum(self.a_lat_max / kappa, eps)))

        # slow down around the first pass of each gate whose true position has not been revealed yet
        d = la.norm(ref.path(s)[:, None, :] - obs["gates_pos"][None, :, :], axis=-1)  # (N, G)
        # gate passes are local distance minima within the gate opening
        is_pass = np.zeros_like(d, dtype=bool)
        is_pass[1:-1] = (d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < pass_tol)

        slow_down = is_pass.any(axis=0) & ~obs["gates_visited"]
        s_gate = s[is_pass.argmax(axis=0)][slow_down]  # progress of the first pass of each gate

        closeness = np.clip(
            (np.abs(s[:, None] - s_gate[None, :]) - self.gate_plateau) / self.gate_window, 0.0, 1.0
        )
        closeness = closeness.min(axis=1, initial=1.0)
        v = np.minimum(v, self.v_gate_max + (self.v_max - self.v_gate_max) * closeness)

        ds = np.diff(s)
        if self.thrust_max is None:
            for i in range(len(v) - 1):  # cap at max acceleration
                v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * self.a_long_max * ds[i]))

            for i in range(len(v) - 2, -1, -1):
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.a_brake_max * ds[i]))
            return self._smooth(s, v)

        tangent = ref.path(s, 1)
        tangent = tangent / np.maximum(la.norm(tangent, axis=-1, keepdims=True), eps)
        curv = ref.path(s, 2)  # curvature vector kappa * normal (arc length parameterization)
        a_t = np.linspace(-self.a_brake_max, self.a_long_max, 121)  # candidate tangential accs

        def feasible(i: np.ndarray, vel: np.ndarray) -> np.ndarray:
            """Feasibility of every candidate tangential acceleration, shape (len(i), 121)."""
            f = (vel[:, None] ** 2 * curv[i])[:, None, :] + a_t[None, :, None] * tangent[i][:, None]
            f[..., 2] += 9.81
            norm = la.norm(f, axis=-1)
            ok = (norm <= self.thrust_max) & (norm >= self.thrust_min)
            if self.tilt_max is not None:
                ok &= f[..., 2] >= norm * np.cos(self.tilt_max)
            return ok

        # Highest speed at every point at which some tangential acceleration is feasible
        idx = np.arange(len(s))
        lo, hi = np.zeros(len(s)), v.copy()
        ok_hi = feasible(idx, hi).any(axis=1)
        for _ in range(25):  # bisection, vectorized over the path
            mid = 0.5 * (lo + hi)
            ok = feasible(idx, mid).any(axis=1)
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        v = np.where(ok_hi, v, lo)

        # Forward / backward passes with the feasible acceleration range at the current speed
        for i in range(len(v) - 1):
            ok = feasible(np.array([i]), v[i : i + 1])[0]
            acc = a_t[ok].max() if ok.any() else 0.0
            v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2 * acc * ds[i], 0.0)))
        for i in range(len(v) - 2, -1, -1):
            ok = feasible(np.array([i + 1]), v[i + 1 : i + 2])[0]
            brake = -a_t[ok].min() if ok.any() else 0.0
            v[i] = min(v[i], np.sqrt(max(v[i + 1] ** 2 + 2 * brake * ds[i], 0.0)))
        return self._smooth(s, v)

    def _smooth(self, s: np.ndarray, v: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Moving average of the moving minimum of the profile over `smooth_time`, in time."""
        if not self.smooth_time:
            return v
        v_safe = np.maximum(v, 0.05)  # the profile starts at rest
        t = np.concatenate(([0.0], np.cumsum(np.diff(s) / (0.5 * (v_safe[1:] + v_safe[:-1])))))
        tt = np.arange(0.0, t[-1] + dt, dt)
        vt = np.interp(tt, t, v)
        half = max(1, int(round(0.5 * self.smooth_time / dt)))
        win = 2 * half + 1
        padded = np.pad(vt, half, mode="edge")
        v_min = np.lib.stride_tricks.sliding_window_view(padded, win).min(axis=1)
        padded = np.pad(v_min, half, mode="edge")
        v_avg = np.lib.stride_tricks.sliding_window_view(padded, win).mean(axis=1)
        # the time axis was built with the original profile; the result is only used as a speed
        # limit at every path point, so map it back through the same time stamps
        return np.minimum(v, np.interp(t, tt, v_avg))
