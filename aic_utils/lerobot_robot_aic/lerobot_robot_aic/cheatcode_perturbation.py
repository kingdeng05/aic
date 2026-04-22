#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

"""Ornstein-Uhlenbeck perturbation for CheatCodeTeleop.

The OU bias on the commanded Cartesian target gradually drifts away from
the true target over a few ticks and is then corrected by the CheatCode
PI loop, reproducing teleop overshoot-and-correct behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transforms3d._gohlketransforms import quaternion_multiply


@dataclass
class PerturbationConfig:
    approach_noise_xyz_m: float = 0.004
    descent_noise_xyz_m: float = 0.001
    approach_rot_noise_deg: float = 2.0
    ou_theta: float = 0.05
    seed: int | None = None


class OUPerturbation:
    """Per-episode OU noise generator."""

    def __init__(self, cfg: PerturbationConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._bias = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self._bias[:] = 0.0

    def perturb_xyz(self, xyz: tuple[float, float, float], phase: str) -> tuple[float, float, float]:
        sigma = (
            self.cfg.approach_noise_xyz_m
            if phase == "approach"
            else self.cfg.descent_noise_xyz_m
        )
        # OU step: bias <- (1-theta)*bias + sigma * N(0, I)
        self._bias = (1.0 - self.cfg.ou_theta) * self._bias + sigma * self._rng.standard_normal(3)
        return (
            float(xyz[0] + self._bias[0]),
            float(xyz[1] + self._bias[1]),
            float(xyz[2] + self._bias[2]),
        )

    def perturb_orientation(
        self, quat_wxyz: tuple[float, float, float, float], phase: str
    ) -> tuple[float, float, float, float]:
        if phase != "approach" or self.cfg.approach_rot_noise_deg <= 0.0:
            return quat_wxyz
        sigma_rad = np.deg2rad(self.cfg.approach_rot_noise_deg)
        axis = self._rng.standard_normal(3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return quat_wxyz
        axis /= norm
        angle = float(sigma_rad * self._rng.standard_normal())
        half = 0.5 * angle
        dq = (
            float(np.cos(half)),
            float(axis[0] * np.sin(half)),
            float(axis[1] * np.sin(half)),
            float(axis[2] * np.sin(half)),
        )
        q = quaternion_multiply(dq, quat_wxyz)
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
