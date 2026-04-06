"""
slic_export.py  —  CIELAB-space SLIC, exports superpixel data as JSON for segment_viewer.html

Usage:
    python slic_export.py <image_path> [output.json]
    python slic_export.py image.png                   # -> image.slic.json
    python slic_export.py image.png out.json --n-segments 400 --compactness 5

Why CIELAB:
    SLIC in CIELAB clusters perceptually uniform colour regions, so superpixel
    boundaries align much better with visible object edges than RGB-based SLIC.
"""

import sys
import json
import base64
import argparse
import io as _io
import numpy as np
from pathlib import Path
from skimage import io, segmentation, color
from skimage.measure import regionprops
from PIL import Image


# ── defaults ──────────────────────────────────────────────────────────────
SLIC_N_SEGMENTS  = 300
SLIC_COMPACTNESS = 10
SLIC_SIGMA       = 1.0


def load_image(path: str) -> np.ndarray:
    """Load any image as float32 RGB in [0, 1]."""
    img = io.imread(path)
    if img.ndim == 2:                        # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:                    # RGBA → RGB
        img = img[:, :, :3]
    return img.astype(np.float32) / 255.0


def image_to_b64(img_f32: np.ndarray) -> str:
    """Encode float32 RGB as a data-URI PNG (embedded in JSON)."""
    img_u8 = (img_f32 * 255).clip(0, 255).astype(np.uint8)
    buf = _io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_superpixels(img_f32: np.ndarray, n_segments, compactness, sigma) -> np.ndarray:
    """SLIC in CIELAB space for perceptually uniform clustering."""
    img_lab = color.rgb2lab(img_f32)          # convert to CIELAB
    return segmentation.slic(
        img_lab,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
        channel_axis=-1,
        convert2lab=False,                    # already in LAB
    )


def export_json(img_f32: np.ndarray, segments: np.ndarray) -> dict:
    H, W = img_f32.shape[:2]
    sp_list = []

    for prop in regionprops(segments):
        sp_id = prop.label
        mask  = segments == sp_id

        mean_color = (img_f32[mask].mean(axis=0) * 255).astype(int).tolist()

        ys, xs = np.where(mask)
        # flat interleaved [y0,x0,y1,x1,...] — smaller JSON than list-of-pairs
        coords = np.empty(len(ys) * 2, dtype=np.int32)
        coords[0::2] = ys
        coords[1::2] = xs

        sp_list.append({
            "id":         sp_id,
            "centroid":   [round(float(prop.centroid[0]), 1),
                           round(float(prop.centroid[1]), 1)],
            "mean_color": mean_color,
            "pixels":     coords.tolist(),
        })

    return {
        "width":       W,
        "height":      H,
        "image_b64":   image_to_b64(img_f32),
        "superpixels": sp_list,
    }


def run(image_path: str, output_json: str = None):
    img_path = Path(image_path)
    if not img_path.exists():
        sys.exit(f"[error] file not found: {img_path}")

    out_path = Path(output_json) if output_json \
               else img_path.with_suffix(".slic.json")

    print(f"[1/3] Loading    {img_path}")
    img_f32 = load_image(str(img_path))
    print(f"      {img_f32.shape[1]}x{img_f32.shape[0]} px")

    print(f"[2/3] SLIC (CIELAB, n={SLIC_N_SEGMENTS}, compactness={SLIC_COMPACTNESS}) ...")
    segments = build_superpixels(img_f32, SLIC_N_SEGMENTS, SLIC_COMPACTNESS, SLIC_SIGMA)
    print(f"      -> {segments.max()} superpixels")

    print(f"[3/3] Exporting  {out_path} ...")
    data = export_json(img_f32, segments)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1e6
    print(f"      -> {size_mb:.1f} MB")
    print(f"[done]  drag {out_path.name} into segment_viewer.html")


# ── entry point — edit paths here ─────────────────────────────────────────
def main():
    IMAGE_PATH  = "wolf.png"
    OUTPUT_JSON = "wolf.slic.json"   # None = same dir as image
    run(IMAGE_PATH, OUTPUT_JSON)


if __name__ == "__main__":
    main()