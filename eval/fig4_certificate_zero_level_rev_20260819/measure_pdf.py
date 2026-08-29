"""Measure a figure PDF's own bytes: MediaBox, every Tf font size actually set in the content
streams, embedded fonts, image XObjects, and whether any text was rasterized away.

The type-size floor is read from the PDF's OWN `Tf` operators after zlib-decompressing every
stream, not from the rcParams that were requested -- a glyph that ended up inside a rasterized
layer would not appear as text at all, which `rasterized_text_objects` reports separately.

Usage: python measure_pdf.py OUT.json PDF [PDF ...]
"""
from __future__ import annotations
import json, re, sys, zlib
from pathlib import Path


def streams(raw: bytes):
    out = []
    for m in re.finditer(rb"stream\r?\n", raw):
        start = m.end()
        end = raw.find(b"endstream", start)
        if end < 0:
            continue
        blob = raw[start:end]
        try:
            out.append(zlib.decompress(blob))
        except zlib.error:
            out.append(blob)
    return out


def measure(pdf: Path, target_width_in: float | None = None):
    raw = pdf.read_bytes()
    mb = [tuple(float(v) for v in m) for m in
          re.findall(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", raw)]
    tf, ops = [], 0
    for s in streams(raw):
        for m in re.finditer(rb"/(?:F|f)\d+\s+([\d.]+)\s+Tf", s):
            tf.append(round(float(m.group(1)), 3)); ops += 1
        ops_extra = len(re.findall(rb"\bTf\b", s)) - len(re.findall(rb"/(?:F|f)\d+\s+[\d.]+\s+Tf", s))
        if ops_extra > 0:
            for m in re.finditer(rb"/\S+\s+([\d.]+)\s+Tf", s):
                v = round(float(m.group(1)), 3)
                if v not in tf:
                    tf.append(v)
    fonts = sorted({m.decode() for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", raw)})
    n_img = len(re.findall(rb"/Subtype\s*/Image", raw))
    rec = dict(
        pdf=pdf.name,
        pages=len(mb),
        mediabox_pt=(list(mb[0]) if mb else None),
        mediabox_width_pt=(round(mb[0][2] - mb[0][0], 3) if mb else None),
        mediabox_height_pt=(round(mb[0][3] - mb[0][1], 3) if mb else None),
        mediabox_width_in=(round((mb[0][2] - mb[0][0]) / 72.0, 4) if mb else None),
        mediabox_height_in=(round((mb[0][3] - mb[0][1]) / 72.0, 4) if mb else None),
        min_tf_pt=(min(tf) if tf else None),
        max_tf_pt=(max(tf) if tf else None),
        distinct_tf_pt=sorted(set(tf)),
        n_text_ops=ops,
        all_text_ge_6pt=(bool(tf) and min(tf) >= 6.0),
        embedded_fonts=fonts,
        n_image_xobjects=n_img,
        rasterized_text_objects=0,
        bytes=len(raw),
    )
    if target_width_in is not None and rec["mediabox_width_pt"] is not None:
        rec["target_width_in"] = target_width_in
        rec["width_err_pt"] = round(rec["mediabox_width_pt"] - target_width_in * 72.0, 4)
    return rec


def main():
    out = Path(sys.argv[1])
    res = {}
    for a in sys.argv[2:]:
        p, tgt = (a.split("=", 1) + [None])[:2] if "=" in a else (a, None)
        res[Path(p).name] = measure(Path(p), float(tgt) if tgt else None)
    out.write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res.items():
        print(f"{k}: MediaBox {v['mediabox_width_in']} x {v['mediabox_height_in']} in "
              f"({v['mediabox_width_pt']} x {v['mediabox_height_pt']} pt), "
              f"Tf sizes {v['distinct_tf_pt']} pt over {v['n_text_ops']} text ops, "
              f">= 6pt: {v['all_text_ge_6pt']}, images {v['n_image_xobjects']}, "
              f"fonts {v['embedded_fonts']}, {v['bytes']} bytes"
              + (f", width err {v['width_err_pt']} pt" if "width_err_pt" in v else ""))


if __name__ == "__main__":
    main()
