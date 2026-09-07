"""Synthetic labeled primitive meshes — a CPU stand-in for ABC while building the
pipeline. Each mesh is a single primitive type, so every face's label is known.
The SAME feature + model + eval code runs on ABC later (where per-face labels come
from ABC's B-rep ground truth); only this loader is swapped out.

Splits are grouped by MESH so no face leaks between train and test.
(`spline` is omitted here — it has no clean synthetic form; ABC supplies real ones.)
"""

from __future__ import annotations

import numpy as np
import trimesh

from tessera.features import face_features
from tessera.validate import genus_of

PRIMITIVES = ["plane", "cylinder", "cone", "sphere", "torus"]
LABEL = {p: i for i, p in enumerate(PRIMITIVES)}


def make_mesh(kind: str, rng: np.random.Generator) -> trimesh.Trimesh:
    if kind == "plane":  # a box: all faces planar
        return trimesh.creation.box(extents=rng.uniform(0.5, 2.0, 3))
    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=int(rng.integers(2, 4)),
                                          radius=float(rng.uniform(0.5, 2.0)))
    if kind == "torus":
        return trimesh.creation.torus(major_radius=float(rng.uniform(1.0, 2.0)),
                                      minor_radius=float(rng.uniform(0.2, 0.5)),
                                      major_sections=int(rng.integers(24, 48)),
                                      minor_sections=int(rng.integers(12, 24)))
    if kind == "cylinder":
        m = trimesh.creation.cylinder(radius=float(rng.uniform(0.3, 1.0)),
                                      height=float(rng.uniform(1.0, 3.0)),
                                      sections=int(rng.integers(24, 48)))
        m.update_faces(np.abs(m.face_normals[:, 2]) < 0.9)  # drop planar caps
        return m
    if kind == "cone":
        m = trimesh.creation.cone(radius=float(rng.uniform(0.5, 1.5)),
                                  height=float(rng.uniform(1.0, 3.0)),
                                  sections=int(rng.integers(24, 48)))
        m.update_faces(m.face_normals[:, 2] > -0.9)  # drop planar base
        return m
    raise ValueError(kind)


def build_dataset(n_per: int = 12, seed: int = 0):
    """Return X (F,8), y (F,), groups (F, mesh id), and per-face slice metadata."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    face_count, genus, prim = [], [], []
    mid = 0
    for kind in PRIMITIVES:
        for _ in range(n_per):
            m = make_mesh(kind, rng)
            if len(m.faces) < 4:
                continue
            f = face_features(m)
            X.append(f)
            y.append(np.full(len(f), LABEL[kind]))
            groups.append(np.full(len(f), mid))
            g = genus_of(m)
            face_count.append(np.full(len(f), len(m.faces)))
            genus.append(np.full(len(f), g if g is not None else -1))
            prim.append(np.full(len(f), LABEL[kind]))
            mid += 1
    return (
        np.vstack(X),
        np.concatenate(y),
        np.concatenate(groups),
        {
            "face_count": np.concatenate(face_count),
            "genus": np.concatenate(genus),
            "primitive": np.concatenate(prim),
        },
    )
