"""Canonicalize a mesh into a pose/scale-invariant frame: center at the centroid,
align principal axes to the world axes (PCA), and scale to the unit bounding
sphere. Deterministic, so the same model always yields the same canonical form."""

from __future__ import annotations

import numpy as np
import trimesh


def canonicalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation(-m.centroid)

    # PCA: rotate so the largest-variance axis maps to x, etc.
    v = np.asarray(m.vertices)
    cov = np.cov(v.T)
    _, evecs = np.linalg.eigh(cov)          # ascending eigenvalues
    R = evecs[:, ::-1]                       # columns = axes, descending variance
    if np.linalg.det(R) < 0:                 # keep it a proper rotation
        R[:, -1] *= -1
    T = np.eye(4)
    T[:3, :3] = R.T
    m.apply_transform(T)

    radius = float(np.linalg.norm(m.vertices, axis=1).max()) or 1.0
    m.apply_scale(1.0 / radius)
    return m
