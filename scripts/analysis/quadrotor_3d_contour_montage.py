"""v2.7.3 M6 — assemble the in-loop 3-D CBF contour frames into a single montage (CPU-only, no GPU).

Reads every figures/inloop/step_*_cbf_contour.png from the M5 run directory in step order and tiles them
into one grid image, so the fixed-recipe contour's evolution over training is a single artifact. Pure PIL;
imports nothing that touches the GPU.
"""
import argparse
import re
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--cols", type=int, default=6)
a = ap.parse_args()

run_dir = Path(a.run_dir)
frames = sorted(
    (run_dir / "figures/inloop").glob("step_*_cbf_contour.png"),
    key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)),
)
if not frames:
    raise SystemExit(f"no contour frames under {run_dir}/figures/inloop")

imgs = [Image.open(p).convert("RGB") for p in frames]
w, h = imgs[0].size
cols = a.cols
rows = (len(imgs) + cols - 1) // cols
montage = Image.new("RGB", (cols * w, rows * h), "white")
for i, im in enumerate(imgs):
    r, c = divmod(i, cols)
    if im.size != (w, h):
        im = im.resize((w, h))
    montage.paste(im, (c * w, r * h))

out = run_dir / "figures" / "cbf_contour_montage.png"
montage.save(out)
steps = [int(re.search(r"step_(\d+)", p.name).group(1)) for p in frames]
print(f"[montage] {len(imgs)} frames (steps {steps[0]}..{steps[-1]}) -> {out} "
      f"grid {cols}x{rows} size {montage.size} bytes {out.stat().st_size}")
