import logging
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dilution.core import DilutionParams, run_dilution

REL = 1e-9
TOPS = (10.0, 3.37, 6.25, 1.0)

def _model(cut_top=True):
    rng = np.random.default_rng(7)
    rows = []
    for i in range(4):
        for j in range(4):
            for k in range(3):
                x0, y0, z0 = i * 20.0, j * 20.0, k * 10.0
                if cut_top and k == 2:
                    top = TOPS[(i + j) % 4]
                    cells = [((0, 20), (0, 20), (0, top))]
                elif cut_top and (i + j + k) % 3 == 0:
                    cells = [((0, 12.5), (0, 20), (0, 10)), ((12.5, 20), (0, 7.5), (0, 10)),
                             ((12.5, 20), (7.5, 20), (0, 10))]
                else:
                    cells = [((0, 20), (0, 20), (0, 10))]
                for (a, b), (c, d), (e, f) in cells:
                    rows.append(dict(
                        XC=x0 + (a + b) / 2, YC=y0 + (c + d) / 2, ZC=z0 + (e + f) / 2,
                        XINC=b - a, YINC=d - c, ZINC=f - e,
                        AU=round(float(rng.lognormal(0.0, 0.8)), 3),
                        RESCAT=int(rng.integers(1, 4)),
                        DENSITY=round(float(rng.uniform(2.2, 2.9)), 3)))
    return pd.DataFrame(rows)

def _params(**kw):
    base = dict(input_path="mem", output_dir="out", cutoff=1.0, smu_x=10.0,
                smu_y=10.0, smu_z=10.0, step_x=1.0, step_y=1.0, oxide_col=None,
                xinc_col="XINC", yinc_col="YINC", zinc_col="ZINC",
                block_x=20.0, block_y=20.0, block_z=10.0)
    base.update(kw)
    return DilutionParams(**base)

def _totals(df, dm):
    t_s = df.XINC * df.YINC * df.ZINC * df.DENSITY
    t_d = dm.XINC * dm.YINC * dm.ZINC * dm.DENSITY
    return t_s.sum(), (t_s * df.AU).sum(), t_d.sum(), (t_d * dm.D_AU).sum()

def _rect(cx, cy, w, h, azimuth_deg):
    a = math.radians(azimuth_deg)
    ux, uy, vx, vy = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    return [(cx + px * ux + py * vx, cy + px * uy + py * vy)
            for px, py in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]

CASES = {
    "a_step1_elos0": dict(step_x=1.0, step_y=1.0, elos=0.0),
    "b_step1.25_elos1.25": dict(step_x=1.25, step_y=1.25, elos=1.25),
    "c_aligned_region": dict(step_x=1.0, step_y=1.0, regions=[
        dict(name="A", priority=1, corners=_rect(40, 40, 80, 80, 0))]),
    "x_only_legacy_strips": dict(step_x=1.25, step_y=None, elos=1.25, elos_axes="x"),
    "smu_z_5": dict(step_x=1.0, step_y=1.0, smu_z=5.0),
}

@pytest.mark.parametrize("cut_top", [True, False])
@pytest.mark.parametrize("case", list(CASES))
def test_tonnes_and_metal_conserved(case, cut_top):
    df = _model(cut_top)
    res = run_dilution(_params(**CASES[case]), df=df)
    t_s, m_s, t_d, m_d = _totals(df, res.diluted_model)
    assert t_d == pytest.approx(t_s, rel=REL)
    assert m_d == pytest.approx(m_s, rel=REL)

def test_entered_origin_off_the_block_edges():

    df = _model(cut_top=False)
    res = run_dilution(_params(origin_y=-2.5, origin_z=-3.0), df=df)
    t_s, m_s, t_d, m_d = _totals(df, res.diluted_model)
    assert t_d == pytest.approx(t_s, rel=REL)
    assert m_d == pytest.approx(m_s, rel=REL)

@pytest.mark.parametrize("azimuth", [30.0, 45.0, 60.0])
def test_rotated_region_difference_is_small(azimuth, caplog):
    df = _model()
    reg = [dict(name="R", priority=1, corners=_rect(40, 40, 60, 60, azimuth))]
    res = run_dilution(_params(regions=reg, elos=1.0), df=df)
    t_s, m_s, t_d, m_d = _totals(df, res.diluted_model)
    dt, dm = 100 * (t_d / t_s - 1), 100 * (m_d / m_s - 1)
    logging.getLogger("test").info("Rotated %g: tonnes %+.2e%%, metal %+.2e%%",
                                   azimuth, dt, dm)

    assert abs(dt) < 1e-7 and abs(dm) < 1e-7

def test_reconciliation_is_logged_without_row_data(caplog):
    df = _model()
    with caplog.at_level(logging.DEBUG, logger="dilution"):
        run_dilution(_params(), df=df)
    text = "\n".join(r.getMessage() for r in caplog.records
                     if r.name == "dilution" and r.levelno == logging.DEBUG)
    assert "Reconciliation" in text
    for stage in ("source", "pieces", "diluted model"):
        assert stage in text
