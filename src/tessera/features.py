"""Per-face geometric features for primitive-type segmentation.

The signal that separates plane / cylinder / cone / sphere / torus is *local
curvature and its anisotropy*: a plane has ~zero dihedral everywhere, a sphere has
uniform curvature (isotropic), a cylinder is curved one way and flat along its axis
(anisotropic). We capture that per face from (a) the dihedral angles to adjacent
faces and (b) the covariance of neighbouring face normals — all CPU, from trimesh.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh

FEATURE_NAMES = [
    "dihedral_mean", "dihedral_std", "dihedral_max", "valence",
    "area_frac", "normal_cov_e0", "normal_cov_e1", "normal_cov_e2",
]


def face_features(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return an (n_faces, 8) feature matrix."""
    n = len(mesh.faces)
    normals = np.asarray(mesh.face_normals)
    areas = np.asarray(mesh.area_faces)
    total = float(areas.sum()) or 1.0

    neigh: dict[int, list[float]] = defaultdict(list)   # face -> dihedral angles
    neigh_faces: dict[int, list[int]] = defaultdict(list)
    adj = np.asarray(mesh.face_adjacency)
    ang = np.asarray(mesh.face_adjacency_angles)
    for (a, b), t in zip(adj, ang):
        neigh[a].append(float(t))
        neigh[b].append(float(t))
        neigh_faces[a].append(b)
        neigh_faces[b].append(a)

    feats = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        angs = np.array(neigh[i]) if neigh[i] else np.zeros(1)
        # covariance eigenvalues of the neighbourhood normals (incl. self)
        idx = neigh_faces[i] + [i]
        nb = normals[idx]
        cov = np.cov(nb.T) if len(idx) > 1 else np.zeros((3, 3))
        ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]  # descending
        s = float(ev.sum()) or 1.0
        feats[i] = [
            angs.mean(), angs.std(), angs.max(), len(angs),
            areas[i] / total, ev[0] / s, ev[1] / s, ev[2] / s,
        ]
    return feats
