"""Each validation rule fires on a mesh engineered to violate exactly it, and a
clean mesh passes all nine. `process=False` keeps trimesh from auto-repairing the
broken meshes before we can test them."""

from __future__ import annotations

import numpy as np
import trimesh

from tessera.validate import genus_of, validate


def _clean():
    return trimesh.creation.icosphere(subdivisions=2)


def test_clean_mesh_passes_all_rules():
    assert validate(_clean()) == []


def test_not_watertight():
    m = _clean()
    keep = np.ones(len(m.faces), bool)
    keep[:2] = False
    m.update_faces(keep)
    assert "NOT_WATERTIGHT" in validate(m)


def test_nan_inf_coords():
    m = _clean().copy()
    v = m.vertices.copy()
    v[0, 0] = np.nan
    m = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)
    assert "NAN_INF_COORDS" in validate(m)


def test_duplicate_vertices():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], float)  # v0 == v3
    f = np.array([[0, 1, 2], [3, 1, 2]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    assert "DUPLICATE_VERTICES" in validate(m)


def test_degenerate_face():
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)  # collinear -> zero area
    m = trimesh.Trimesh(vertices=v, faces=np.array([[0, 1, 2]]), process=False)
    assert "DEGENERATE_FACE" in validate(m)


def test_non_manifold_edge():
    # edge (0,1) shared by three faces
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    assert "NON_MANIFOLD_EDGE" in validate(m)


def test_inconsistent_normals():
    m = _clean().copy()
    f = m.faces.copy()
    f[0] = f[0][::-1]  # flip one winding
    m = trimesh.Trimesh(vertices=m.vertices, faces=f, process=False)
    assert "INCONSISTENT_NORMALS" in validate(m)


def test_unit_scale_implausible():
    m = _clean().copy()
    m.apply_scale(1e6)  # extents ~2e6 -> outside plausible unit range
    assert "UNIT_SCALE_IMPLAUSIBLE" in validate(m)


def test_genus_of_torus_is_one():
    t = trimesh.creation.torus(major_radius=2.0, minor_radius=0.5)
    assert t.is_watertight
    assert genus_of(t) == 1
    assert "GENUS_ANOMALY" not in validate(t)


def test_self_intersection_detected():
    # triangle A in z=0 plane; triangle B pierces through A's interior
    va = np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0]], float)
    vb = np.array([[1, -1, -1], [1, 1, -1], [1, 0, 1]], float)
    v = np.vstack([va, vb])
    f = np.array([[0, 1, 2], [3, 4, 5]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    assert "SELF_INTERSECTION" in validate(m)
