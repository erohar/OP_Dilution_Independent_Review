import logging
import sys

import numpy as np
import pandas as pd

def make_model(nx=120, ny=60, nz=30, bx=5.0, by=5.0, bz=5.0, seed=1):
    rng = np.random.default_rng(seed)
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    x = 10000 + (ix + 0.5) * bx
    y = 20000 + (iy + 0.5) * by
    z = 200 + (iz + 0.5) * bz

    au = np.zeros(ix.shape)
    for x_mid, width, dip in ((180, 12, 0.6), (330, 6, -0.4), (470, 20, 0.2)):
        centre = 10000 + x_mid + dip * (z - 275) + 10 * np.sin(y / 60)
        inside = np.abs(x - centre) < width / 2 + rng.normal(0, 3, ix.shape)
        au += np.where(inside, rng.lognormal(0.2, 0.8, ix.shape), 0)
    au += rng.lognormal(-3.5, 1.0, ix.shape)

    dist = np.hypot((y - 20150) / 150, (z - 330) / 80)
    rescat = np.select([dist < 0.6, dist < 1.0, dist < 1.4], [1, 2, 3], 4)
    dens = np.where(au > 0.3, 2.8, 2.65)
    return pd.DataFrame({
        "XC": x.ravel(), "YC": y.ravel(), "ZC": z.ravel(),
        "AU": np.round(au.ravel(), 3), "RESCAT": rescat.ravel(),
        "DENSITY": dens.ravel(),
    })

log = logging.getLogger("make_synthetic_model")

def main(argv) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = argv[0] if argv else "synthetic_block_model.csv"
    df = make_model()
    try:
        df.to_csv(out, index=False)
    except OSError as e:
        log.error("Could not write %s: %s", out, e)
        return 1
    log.info("Wrote %s blocks to %s", f"{len(df):,}", out)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
