from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .core import (DEFAULT_NAME, OUTSIDE, DilutionParams, _explode, block_extents,
                   definition_grade, infer_block_size, model_grid)
from .regions import Region, as_regions, order_regions, owner
from .store import as_store
from . import theme as _theme

log = logging.getLogger(__name__)

PALETTE = _theme.SERIES
OUTSIDE_COLOUR = _theme.CHART_OUTSIDE
MODEL_COLOUR = _theme.PRINT_MODEL

def region_colour(i: int) -> str:
    return OUTSIDE_COLOUR if i < 0 else PALETTE[i % len(PALETTE)]

def plan_grid(src, p: DilutionParams, max_cells: int = 600,
              progress: Optional[Callable[[str], None]] = None) -> dict:
    store = as_store(src, p, progress)
    try:
        return _plan_grid(store, p, max_cells)
    finally:
        if src is None:
            store.close()

def _plan_grid(store, p: DilutionParams, max_cells: int) -> dict:
    parent = infer_block_size(store, p)
    grid = model_grid(store, p, parent)
    xlo, xhi, ylo, yhi = grid["xlo"], grid["xhi"], grid["ylo"], grid["yhi"]
    span = max(xhi - xlo, yhi - ylo, 1e-9)
    res = span / max_cells
    nx = max(1, int(math.ceil((xhi - xlo) / res - 1e-9)))
    ny = max(1, int(math.ceil((yhi - ylo) / res - 1e-9)))
    tonnes = np.zeros(nx * ny)
    ore = np.zeros(nx * ny)
    metal = np.zeros(nx * ny)
    for sl in store.chunks():
        e = block_extents(store, p, parent, sl)
        if not len(e["X"]):
            continue
        vol = e["xs"] * e["ys"] * e["zs"]
        if p.density_col:
            d = np.asarray(store.num(p.density_col)[sl], float)
            d = np.where(np.isfinite(d), d, p.default_density)
            d = np.where(d > 0, d, 0.0)
        else:
            d = np.full(len(vol), p.default_density)
        t = vol * d
        g = definition_grade(np.asarray(store.num(p.grade_col)[sl], float),
                             np.asarray(store.num(p.rescat_col)[sl], float), p.mode)
        xr, ix, fx = _explode(e["X"] - e["xs"] / 2, e["X"] + e["xs"] / 2, xlo, res)
        yr, iy, fy = _explode(e["Y"][xr] - e["ys"][xr] / 2,
                              e["Y"][xr] + e["ys"][xr] / 2, ylo, res)
        b = xr[yr]
        cell = np.clip(iy, 0, ny - 1) * nx + np.clip(ix[yr], 0, nx - 1)
        w = t[b] * fx[yr] * fy
        tonnes += np.bincount(cell, weights=w, minlength=nx * ny)
        above = g[b] >= p.cutoff
        ore += np.bincount(cell[above], weights=w[above], minlength=nx * ny)
        metal += np.bincount(cell[above], weights=(w * g[b])[above], minlength=nx * ny)
    return dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, res=res, nx=nx, ny=ny,
                tonnes=tonnes.reshape(ny, nx), ore=ore.reshape(ny, nx),
                metal=metal.reshape(ny, nx), cutoff=p.cutoff, mode=p.mode,
                grade_col=p.grade_col)

def assign_cells(plan: dict, regions: list[Region]) -> np.ndarray:
    xs = plan["xlo"] + (np.arange(plan["nx"]) + 0.5) * plan["res"]
    ys = plan["ylo"] + (np.arange(plan["ny"]) + 0.5) * plan["res"]
    X, Y = np.meshgrid(xs, ys)
    return owner(X, Y, regions)

def region_table(plan: dict, regions: list[Region], outside: str) -> pd.DataFrame:
    who = assign_cells(plan, regions)
    rows = []
    for i, r in enumerate(regions + [None]):
        sel = who == (i if r is not None else -1)
        t = float(plan["ore"][sel].sum())
        mt = float(plan["metal"][sel].sum())
        rows.append(dict(
            REGION=r.name if r is not None else DEFAULT_NAME,
            PRIORITY=r.priority if r is not None else None,
            AZIMUTH=round(r.azimuth, 1) if r is not None else None,
            LENGTH=round(r.length, 1) if r is not None else None,
            WIDTH=round(r.width, 1) if r is not None else None,
            ORE_TONNES=t, ORE_GRADE=mt / t if t > 0 else 0.0,
            NOTE="" if r is not None else f"outside every region: {OUTSIDE[outside]}"))
    return pd.DataFrame(rows)

def runs(mask: np.ndarray):
    out = []
    for j in range(mask.shape[0]):
        row = mask[j].astype(np.int8)
        d = np.diff(np.concatenate([[0], row, [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            out.append((j, int(a), int(b)))
    return out

def plan_layers(plan: dict, regions: list[Region], model_colour: str = MODEL_COLOUR
                ) -> list[tuple[str, str, list]]:
    who = assign_cells(plan, regions)
    has = plan["tonnes"] > 0
    ore = plan["ore"] > 0
    layers = [("model below cut-off", model_colour, runs(has & ~ore))]
    for i, r in enumerate(regions):
        layers.append((f"ore in {r.name}", region_colour(i), runs(ore & (who == i))))
    layers.append(("ore outside every region", OUTSIDE_COLOUR, runs(ore & (who < 0))))
    return layers

def arrow(r: Region) -> tuple[tuple[float, float], tuple[float, float]]:
    u0, u1 = r.length / 3, 2 * r.length / 3
    v = r.width / 2
    x, y = r.to_world([u0, u1], [v, v])
    return (float(x[0]), float(y[0])), (float(x[1]), float(y[1]))

def write_pdf(path: str, plan: dict, regions: list[Region], p: DilutionParams,
              title: str = "Dilution regions preview") -> str:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A3, landscape
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError as e:
        raise RuntimeError("The PDF preview needs reportlab: pip install reportlab") from e
    regions = order_regions(as_regions(regions))
    W, H = landscape(A3)
    c = rl_canvas.Canvas(path, pagesize=(W, H))
    c.setTitle(title)
    margin = 36
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - margin - 6, title)
    c.setFont("Helvetica", 9)
    c.drawString(margin, H - margin - 22,
                 f"{p.input_path}   |   ore = {p.grade_col}_{p.mode} >= {p.cutoff:g}, "
                 f"all benches   |   {_dt.datetime.now():%Y-%m-%d %H:%M}")

    table_w = 330
    ax0, ay0 = margin, margin
    aw, ah = W - 3 * margin - table_w, H - 2 * margin - 40
    xlo, ylo, res = plan["xlo"], plan["ylo"], plan["res"]
    wx, wy = plan["nx"] * res, plan["ny"] * res
    bx = [xlo, xlo + wx]
    by = [ylo, ylo + wy]
    for r in regions:
        x0_, x1_, y0_, y1_ = r.bbox()
        bx, by = [min(bx[0], x0_), max(bx[1], x1_)], [min(by[0], y0_), max(by[1], y1_)]
    s = min(aw / max(bx[1] - bx[0], 1e-9), ah / max(by[1] - by[0], 1e-9))
    ay0 += ah - (by[1] - by[0]) * s

    def px(x):
        return ax0 + (x - bx[0]) * s

    def py(y):
        return ay0 + (y - by[0]) * s

    c.setStrokeColor(colors.HexColor(_theme.PRINT_FRAME))
    c.rect(px(bx[0]), py(by[0]), (bx[1] - bx[0]) * s, (by[1] - by[0]) * s)
    for _, colour, rr in plan_layers(plan, regions):
        c.setFillColor(colors.HexColor(colour))
        for j, a, b in rr:

            c.rect(px(xlo + a * res), py(ylo + j * res), (b - a) * res * s,
                   res * s + 0.3, stroke=0, fill=1)
    for i, r in enumerate(regions):
        col = colors.HexColor(region_colour(i))
        poly = r.polygon()
        c.setStrokeColor(col)
        c.setLineWidth(1.6)
        outline = c.beginPath()
        outline.moveTo(px(poly[0, 0]), py(poly[0, 1]))
        for x, y in poly[1:]:
            outline.lineTo(px(x), py(y))
        outline.close()
        c.drawPath(outline, stroke=1, fill=0)
        (x0_, y0_), (x1_, y1_) = arrow(r)
        _pdf_arrow(c, px(x0_), py(y0_), px(x1_), py(y1_))
        cx, cy = r.to_world(r.length / 2, r.width * 0.7)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(px(float(cx)), py(float(cy)),
                            f"{r.name} (P{r.priority})")

    tx, ty = W - margin - table_w, H - margin - 50
    c.setFont("Helvetica-Bold", 10)
    c.drawString(tx, ty, "Legend")
    ty -= 14
    c.setFont("Helvetica", 8.5)
    for label, colour, _ in plan_layers(plan, regions)[:1] + [
            (f"ore in {r.name}", region_colour(i), None) for i, r in enumerate(regions)] + [
            ("ore outside every region", OUTSIDE_COLOUR, None)]:
        c.setFillColor(colors.HexColor(colour))
        c.rect(tx, ty - 2, 10, 8, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(tx + 14, ty, label)
        ty -= 12
    c.drawString(tx, ty, "Arrow: dilution direction (region corner 1 to 2), across strike")
    ty -= 22
    tab = region_table(plan, regions, p.outside)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(tx, ty, "Regions (claim order)")
    ty -= 14
    c.setFont("Helvetica-Bold", 8)
    heads = [("Region", 0), ("Prio", 90), ("Azimuth", 120), ("L x W (m)", 165),
             ("Ore t", 245), ("Grade", 295)]
    for h, off in heads:
        c.drawString(tx + off, ty, h)
    ty -= 12
    c.setFont("Helvetica", 8)
    for row in tab.itertuples(index=False):
        vals = [str(row.REGION)[:16],
                "" if pd.isna(row.PRIORITY) else f"{int(row.PRIORITY)}",
                "" if pd.isna(row.AZIMUTH) else f"{row.AZIMUTH:05.1f}",
                "" if pd.isna(row.LENGTH) else f"{row.LENGTH:,.0f} x {row.WIDTH:,.0f}",
                f"{row.ORE_TONNES:,.0f}", f"{row.ORE_GRADE:.2f}"]
        for (_, off), v in zip(heads, vals):
            c.drawString(tx + off, ty, v)
        ty -= 11
        if row.NOTE:
            c.setFont("Helvetica-Oblique", 7.5)
            c.drawString(tx + 8, ty, row.NOTE)
            c.setFont("Helvetica", 8)
            ty -= 11
    ty -= 8
    c.setFont("Helvetica", 7.5)
    for line in ("Ore tonnes are summed on a plan grid of "
                 f"{res:,.2f} m, so they are approximate.",
                 "Where regions overlap, the lower priority number dilutes the overlap."):
        c.drawString(tx, ty, line)
        ty -= 10
    c.showPage()
    c.save()
    return path

def _pdf_arrow(c, x0: float, y0: float, x1: float, y1: float) -> None:
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1.2)
    c.line(x0, y0, x1, y1)
    ang = math.atan2(y1 - y0, x1 - x0)
    for d in (2.6, -2.6):
        c.line(x1, y1, x1 - 8 * math.cos(ang + d / 6), y1 - 8 * math.sin(ang + d / 6))

def region_preview(src, p: DilutionParams, path: str,
                   progress: Optional[Callable[[str], None]] = None) -> tuple[str, pd.DataFrame]:
    regions = order_regions(as_regions(p.regions))
    plan = plan_grid(src, p, progress=progress)
    write_pdf(path, plan, regions, p)
    return path, region_table(plan, regions, p.outside)
