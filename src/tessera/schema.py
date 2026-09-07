"""Typed records + error taxonomy for the geometry pipeline.

Validation failures are an *enum*, not a boolean — so a quarantined model can be
routed, counted per class, and debugged. This taxonomy is what makes Phase 5's
slice-based eval possible (you can slice by failure mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ErrorClass(str, Enum):
    NAN_INF_COORDS = "NAN_INF_COORDS"
    DEGENERATE_FACE = "DEGENERATE_FACE"
    DUPLICATE_VERTICES = "DUPLICATE_VERTICES"
    NON_MANIFOLD_EDGE = "NON_MANIFOLD_EDGE"
    NOT_WATERTIGHT = "NOT_WATERTIGHT"
    INCONSISTENT_NORMALS = "INCONSISTENT_NORMALS"
    SELF_INTERSECTION = "SELF_INTERSECTION"  # broad-phase screen (see validate.py)
    GENUS_ANOMALY = "GENUS_ANOMALY"
    UNIT_SCALE_IMPLAUSIBLE = "UNIT_SCALE_IMPLAUSIBLE"


@dataclass
class GeometryRecord:
    """One processed model: provenance + validation outcome + basic geometry stats."""

    source_path: str
    content_hash: str
    n_vertices: int
    n_faces: int
    bbox_extent: float
    is_valid: bool
    errors: list[ErrorClass] = field(default_factory=list)
    genus: int | None = None

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "bbox_extent": round(self.bbox_extent, 6),
            "is_valid": self.is_valid,
            "errors": [e.value if isinstance(e, ErrorClass) else str(e) for e in self.errors],
            "genus": self.genus,
        }
