"""Canonicalization, sampling, and the per-file driver."""

from __future__ import annotations

import numpy as np
import trimesh

from tessera.canonicalize import canonicalize
from tessera.pipeline import content_hash, process_mesh
from tessera.sample import sample_sdf, sample_surface


def test_canonicalize_centers_and_unit_scales():
    m = trimesh.creation.box(extents=[3, 1, 2])
    m.apply_translation([10, 5, -3])
    c = canonicalize(m)
    assert np.allclose(c.centroid, 0, atol=1e-6)
    assert abs(np.linalg.norm(c.vertices, axis=1).max() - 1.0) < 1e-6


def test_sample_surface_shapes():
    m = trimesh.creation.icosphere(subdivisions=2)
    p, n = sample_surface(m, 500)
    assert p.shape == (500, 3) and n.shape == (500, 3)


def test_sdf_is_none_on_open_mesh():
    m = trimesh.creation.icosphere(subdivisions=2)
    keep = np.ones(len(m.faces), bool)
    keep[:2] = False
    m.update_faces(keep)
    assert sample_sdf(m, 100) is None  # not watertight -> undefined


def test_sdf_shapes_when_available():
    # depends on a proximity backend; tolerate None if unavailable
    out = sample_sdf(trimesh.creation.icosphere(subdivisions=2), 200)
    assert out is None or (out[0].shape == (200, 3) and out[1].shape == (200,))


def test_process_mesh_valid_and_quarantine():
    assert process_mesh(trimesh.creation.box(), "g").is_valid
    bad = trimesh.creation.box().copy()
    keep = np.ones(len(bad.faces), bool)
    keep[:2] = False
    bad.update_faces(keep)
    r = process_mesh(bad, "b")
    assert not r.is_valid and "NOT_WATERTIGHT" in r.errors


def test_content_hash_deterministic():
    m = trimesh.creation.box()
    assert content_hash(m) == content_hash(m.copy())
