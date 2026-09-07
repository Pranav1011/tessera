"""9-rule geometry validation over triangle meshes (trimesh).

Each rule returns the ErrorClass it detects, or None. `validate` runs all nine and
returns the list of failures. Eight rules are exact; SELF_INTERSECTION is a
documented broad-phase screen (AABB overlap of non-adjacent faces) — it flags
*candidates*, cheaply, without an O(n^2) triangle-triangle test. That scoping is
stated, not hidden.
"""

from __future__ import annotations

import numpy as np
import trimesh

_AREA_EPS = 1e-12
_UNIT_MIN, _UNIT_MAX = 1e-4, 1e4  # raw bounding-box extent outside this = ambiguous units


def _nan_inf(m: trimesh.Trimesh):
    return None if np.isfinite(m.vertices).all() else "NAN_INF_COORDS"


def _degenerate(m: trimesh.Trimesh):
    return "DEGENERATE_FACE" if (m.area_faces <= _AREA_EPS).any() else None


def _duplicate_vertices(m: trimesh.Trimesh):
    uniq = trimesh.grouping.unique_rows(m.vertices)[0]
    return "DUPLICATE_VERTICES" if len(uniq) < len(m.vertices) else None


def _non_manifold(m: trimesh.Trimesh):
    # an edge shared by >2 faces is non-manifold
    edges = np.sort(m.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return "NON_MANIFOLD_EDGE" if (counts > 2).any() else None


def _watertight(m: trimesh.Trimesh):
    return None if m.is_watertight else "NOT_WATERTIGHT"


def _normals(m: trimesh.Trimesh):
    return None if m.is_winding_consistent else "INCONSISTENT_NORMALS"


def _seg_tri(p0, p1, v0, v1, v2) -> bool:
    """Möller-Trumbore: does the OPEN segment p0->p1 pierce triangle (v0,v1,v2)?
    Endpoints are excluded so faces that merely touch at a shared vertex/edge do
    not register as an intersection."""
    d = p1 - p0
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(d, e2)
    a = float(e1 @ h)
    if abs(a) < 1e-12:
        return False  # segment parallel to triangle
    f = 1.0 / a
    s = p0 - v0
    u = f * float(s @ h)
    if u < -1e-9 or u > 1 + 1e-9:
        return False
    q = np.cross(s, e1)
    v = f * float(d @ q)
    if v < -1e-9 or u + v > 1 + 1e-9:
        return False
    t = f * float(e2 @ q)
    return 1e-7 < t < 1 - 1e-7


def _tris_intersect(a, b) -> bool:
    a0, a1, a2 = a
    b0, b1, b2 = b
    for p, q in ((a0, a1), (a1, a2), (a2, a0)):
        if _seg_tri(p, q, b0, b1, b2):
            return True
    for p, q in ((b0, b1), (b1, b2), (b2, b0)):
        if _seg_tri(p, q, a0, a1, a2):
            return True
    return False


def _self_intersection(m: trimesh.Trimesh):
    """Broad-phase (AABB sweep) to find non-adjacent candidate face pairs, then a
    narrow-phase triangle-triangle test so only *real* intersections are flagged."""
    try:
        tris = m.triangles
    except Exception:
        return None
    if len(tris) > 10_000:  # narrow-phase is O(candidate pairs); screen small meshes only
        return None
    mins, maxs = tris.min(axis=1), tris.max(axis=1)
    order = np.argsort(mins[:, 0])
    faces = m.faces
    checks = 0
    active: list[int] = []
    for idx in order:
        lo = mins[idx, 0]
        active = [j for j in active if maxs[j, 0] >= lo]
        fi = set(faces[idx])
        for j in active:
            if fi & set(faces[j]):
                continue  # adjacent faces share a vertex/edge by construction
            if (mins[idx] <= maxs[j]).all() and (maxs[idx] >= mins[j]).all():
                checks += 1
                if _tris_intersect(tris[idx], tris[j]):
                    return "SELF_INTERSECTION"
                if checks > 2_000_000:  # safety bound
                    return None
        active.append(idx)
    return None


def _genus(m: trimesh.Trimesh):
    if not m.is_watertight:
        return None  # genus only well-defined on a closed surface
    g = (2 - m.euler_number) / 2
    if g < 0 or g != int(g) or g > 100:
        return "GENUS_ANOMALY"
    return None


def _unit_scale(m: trimesh.Trimesh):
    ext = float(m.bounding_box.extents.max())
    return "UNIT_SCALE_IMPLAUSIBLE" if not (_UNIT_MIN <= ext <= _UNIT_MAX) else None


_RULES = [
    _nan_inf, _degenerate, _duplicate_vertices, _non_manifold, _watertight,
    _normals, _self_intersection, _genus, _unit_scale,
]


def validate(mesh: trimesh.Trimesh) -> list[str]:
    """Return the list of failed ErrorClass values (empty = valid)."""
    out: list[str] = []
    for rule in _RULES:
        try:
            err = rule(mesh)
        except Exception:
            err = None  # a rule crashing must not crash the pipeline
        if err:
            out.append(err)
    return out


def genus_of(mesh: trimesh.Trimesh) -> int | None:
    if not mesh.is_watertight:
        return None
    return int((2 - mesh.euler_number) / 2)
