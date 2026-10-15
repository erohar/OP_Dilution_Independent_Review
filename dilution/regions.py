from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MAX_SKEW_DEG = 1.0
_MAX_CLOSURE = 0.01

DEFAULT_NAME = "MODEL"

@dataclass(frozen=True)
class Region:
    name: str
    corners: tuple
    priority: int = 1

    @property
    def origin(self) -> tuple[float, float]:
        return float(self.corners[0][0]), float(self.corners[0][1])

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        p1, p2, p3 = (np.asarray(c, float) for c in self.corners[:3])
        u = p2 - p1
        u = u / np.hypot(*u)
        v = np.array([-u[1], u[0]])
        if np.dot(p3 - p2, v) < 0:
            v = -v
        return u, v

    @property
    def length(self) -> float:
        p1, p2 = (np.asarray(c, float) for c in self.corners[:2])
        return float(np.hypot(*(p2 - p1)))

    @property
    def width(self) -> float:
        _, v = self.axes
        p2, p3 = (np.asarray(c, float) for c in self.corners[1:3])
        return float(abs(np.dot(p3 - p2, v)))

    @property
    def azimuth(self) -> float:
        u, _ = self.axes
        return math.degrees(math.atan2(u[0], u[1])) % 360.0

    def to_local(self, x, y):
        (ox, oy), (u, v) = self.origin, self.axes
        dx, dy = np.asarray(x, float) - ox, np.asarray(y, float) - oy
        return dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]

    def to_world(self, uu, vv):
        (ox, oy), (u, v) = self.origin, self.axes
        uu, vv = np.asarray(uu, float), np.asarray(vv, float)
        return ox + uu * u[0] + vv * v[0], oy + uu * u[1] + vv * v[1]

    def contains(self, x, y) -> np.ndarray:
        uu, vv = self.to_local(x, y)
        return (uu >= 0) & (uu < self.length) & (vv >= 0) & (vv < self.width)

    def polygon(self) -> np.ndarray:
        uu = np.array([0.0, self.length, self.length, 0.0])
        vv = np.array([0.0, 0.0, self.width, self.width])
        x, y = self.to_world(uu, vv)
        return np.column_stack([x, y])

    def bbox(self) -> tuple[float, float, float, float]:
        poly = self.polygon()
        return (float(poly[:, 0].min()), float(poly[:, 0].max()),
                float(poly[:, 1].min()), float(poly[:, 1].max()))

    def touches_boxes(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        xmin, xmax = np.asarray(xmin, float), np.asarray(xmax, float)
        ymin, ymax = np.asarray(ymin, float), np.asarray(ymax, float)
        poly = self.polygon()
        tol = 1e-9
        hit = ((xmax > poly[:, 0].min() + tol) & (xmin < poly[:, 0].max() - tol)
               & (ymax > poly[:, 1].min() + tol) & (ymin < poly[:, 1].max() - tol))
        (ox, oy), (u, v) = self.origin, self.axes
        for axis, hi in ((u, self.length), (v, self.width)):

            p = [((cx - ox) * axis[0] + (cy - oy) * axis[1])
                 for cx in (xmin, xmax) for cy in (ymin, ymax)]
            lo_p = np.minimum.reduce(p)
            hi_p = np.maximum.reduce(p)
            hit &= (hi_p > tol) & (lo_p < hi - tol)
        return hit

    def overlaps(self, other: "Region") -> bool:
        a, b = self.polygon(), other.polygon()
        for poly in (a, b):
            for i in range(4):
                edge = poly[(i + 1) % 4] - poly[i]
                n = np.array([-edge[1], edge[0]])
                pa, pb = a @ n, b @ n
                if pa.max() <= pb.min() + 1e-9 or pb.max() <= pa.min() + 1e-9:
                    return False
        return True

    def describe(self) -> str:
        return (f"{self.name}: priority {self.priority}, {self.length:,.1f} m "
                f"along azimuth {self.azimuth:05.1f} by {self.width:,.1f} m")

def make_region(name, points, priority: int = 1) -> Region:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) == 5 and math.dist(pts[0], pts[4]) < 1e-6:
        pts = pts[:4]
    if len(pts) != 4:
        raise ValueError(f"Region {name}: needs 4 corner points, got {len(pts)}")
    for i in range(4):
        for j in range(i + 1, 4):
            if math.dist(pts[i], pts[j]) < 1e-6:
                raise ValueError(f"Region {name}: corners {i + 1} and {j + 1} "
                                 "are the same point")
    p = np.asarray(pts)
    s1, s2 = p[1] - p[0], p[2] - p[1]
    cos = abs(float(np.dot(s1, s2))) / (np.hypot(*s1) * np.hypot(*s2))
    skew = 90.0 - math.degrees(math.acos(min(1.0, cos)))
    closure = float(np.hypot(*(p[0] + s2 - p[3]))) / float(np.hypot(*(p[2] - p[0])))
    if skew > _MAX_SKEW_DEG or closure > _MAX_CLOSURE:
        raise ValueError(
            f"Region {name}: the 4 points are not a rectangle (corner 2 is "
            f"{90 - skew:.2f} degrees and corner 4 is {closure:.1%} of the "
            "diagonal from where a rectangle puts it). Check the point order: "
            "corner 1 to 2 is the dilution direction, 2 to 3 along strike")
    if skew > 0.01 or closure > 1e-4:
        log.warning("Region %s is %.2f degrees out of square; it is squared "
                    "up on its first side", name, skew)
    return Region(str(name), tuple(pts), int(priority))

def order_regions(regions: list[Region]) -> list[Region]:
    regions = list(regions)
    names = [r.name for r in regions]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Region names must be unique: {', '.join(dup)}")
    if any(r.name == DEFAULT_NAME for r in regions):
        raise ValueError(f"'{DEFAULT_NAME}' is reserved for material outside "
                         "every region; rename that region")
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            if a.priority == b.priority and a.overlaps(b):
                raise ValueError(
                    f"Regions {a.name} and {b.name} overlap and both have "
                    f"priority {a.priority}. Give the one whose dilution should "
                    "apply in the overlap the lower priority number")
    return sorted(regions, key=lambda r: (r.priority, names.index(r.name)))

def owner(x, y, regions: list[Region]) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    shape = x.shape
    x, y = x.ravel(), y.ravel()
    out = np.full(x.shape, -1, dtype=np.int64)
    for i, r in enumerate(regions):
        free = out < 0
        if not free.any():
            break
        x0, x1, y0, y1 = r.bbox()
        cand = free & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if cand.any():
            idx = np.flatnonzero(cand)
            out[idx[r.contains(x[idx], y[idx])]] = i
    return out.reshape(shape)

_ALIASES = {"REGION": ("REGION", "NAME", "ID", "PVALUE", "STRING", "REGION_ID"),
            "PRIORITY": ("PRIORITY", "PRIO", "RANK"),
            "X": ("X", "XP", "EAST", "EASTING"),
            "Y": ("Y", "YP", "NORTH", "NORTHING")}

def _find(columns, key) -> Optional[str]:
    upper = {str(c).strip().upper(): c for c in columns}
    return next((upper[a] for a in _ALIASES[key] if a in upper), None)

def _priority(value, default: int) -> int:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    if not v.is_integer():
        raise ValueError(f"Region priority must be a whole number, got {value}")
    return int(v)

def regions_from_frame(df: pd.DataFrame) -> list[Region]:
    cols = {str(c).strip().upper(): c for c in df.columns}
    name_col, prio_col = _find(df.columns, "REGION"), _find(df.columns, "PRIORITY")
    wide = all(f"{a}{i}" in cols for a in "XY" for i in range(1, 5))
    out = []
    if wide:
        for n, row in enumerate(df.itertuples(index=False), 1):
            r = {str(k).strip().upper(): v for k, v in zip(df.columns, row)}
            name = r.get(str(name_col).strip().upper()) if name_col else n
            pts = [(r[f"X{i}"], r[f"Y{i}"]) for i in range(1, 5)]
            out.append(make_region(_label(name), pts, _priority(
                r.get(str(prio_col).strip().upper()) if prio_col else None, n)))
        return out
    x_col, y_col = _find(df.columns, "X"), _find(df.columns, "Y")
    if not (name_col and x_col and y_col):
        raise ValueError("The regions file needs REGION, X1, Y1 ... X4, Y4 columns "
                         "(one row per region) or REGION, X, Y (one row per point)")
    for n, (name, g) in enumerate(df.groupby(name_col, sort=False), 1):
        pr = _priority(g[prio_col].iloc[0], n) if prio_col else n
        out.append(make_region(_label(name), list(zip(g[x_col], g[y_col])), pr))
    return out

def _label(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()

def read_surpac_regions(path: str) -> list[Region]:
    pts: dict = {}
    prio: dict = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    for line in lines[2:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            sn = int(float(parts[0]))
            y, x = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if sn == 0:
            continue
        pts.setdefault(sn, []).append((x, y))
        if sn not in prio and len(parts) > 4 and parts[4]:
            prio[sn] = parts[4]
    if not pts:
        raise ValueError(f"No strings found in {path}")
    return [make_region(str(sn), p, _priority(prio.get(sn), i))
            for i, (sn, p) in enumerate(pts.items(), 1)]

def load_regions(path: str) -> list[Region]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Regions file not found: {path}")
    if os.path.splitext(path)[1].lower() == ".str":
        regions = read_surpac_regions(path)
    else:
        try:
            df = pd.read_csv(path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            raise ValueError(f"Could not read the regions file {path}: {e}") from e
        regions = regions_from_frame(df)
    if not regions:
        raise ValueError(f"No regions in {path}")
    order_regions(regions)
    return regions

def regions_to_frame(regions: list[Region]) -> pd.DataFrame:
    rows = []
    for r in regions:
        row = {"REGION": r.name, "PRIORITY": r.priority}
        for i, (x, y) in enumerate(r.corners, 1):
            row[f"X{i}"], row[f"Y{i}"] = x, y
        row.update(AZIMUTH=round(r.azimuth, 3), LENGTH=r.length, WIDTH=r.width)
        rows.append(row)
    return pd.DataFrame(rows)

def as_regions(items) -> list[Region]:
    out = []
    for i, r in enumerate(items or [], 1):
        if isinstance(r, Region):
            out.append(r)
        else:
            out.append(make_region(r.get("name", i), r["corners"],
                                   r.get("priority", i)))
    return out
