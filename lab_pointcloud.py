"""
lab_pointcloud.py
-----------------
Reads segmentation_labels.json + original image,
plots foreground vs background pixels in CIE a*b* space.

Usage:
    python lab_pointcloud.py <image_path> <labels_json> [output_png]
"""

import sys
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from skimage import io, color


def load_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.float32) / 255.0


def decode_mask(b64: str, width: int, height: int) -> np.ndarray:
    """Decode pixel_mask_b64 -> 2D array: 0=bg, 1=fg, 2=uncertain"""
    raw = base64.b64decode(b64)
    flat = np.frombuffer(raw, dtype=np.uint8)
    return flat.reshape(height, width)


def plot(img_f32, mask, out_path):
    H, W = img_f32.shape[:2]
    lab = color.rgb2lab(img_f32)          # H×W×3, channels: L, a*, b*
    a   = lab[:, :, 1].ravel()            # green–red axis
    b   = lab[:, :, 2].ravel()            # blue–yellow axis
    flat_mask = mask.ravel()

    fg_idx = flat_mask == 1
    bg_idx = flat_mask == 0
    un_idx = flat_mask == 2

    rgb_flat = img_f32.reshape(-1, 3)

    # ── matplotlib academic style ──────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         9,
        "axes.linewidth":    0.8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#aaa",
    })

    fig = plt.figure(figsize=(13, 4.4), facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                            left=0.07, right=0.97, top=0.84, bottom=0.14)

    # ── shared axis limits ─────────────────────────────────────────────
    a_all = np.concatenate([a[fg_idx], a[bg_idx]])
    b_all = np.concatenate([b[fg_idx], b[bg_idx]])
    pad   = 6
    alim  = (a_all.min() - pad, a_all.max() + pad)
    blim  = (b_all.min() - pad, b_all.max() + pad)

    def style_ax(ax):
        ax.set_facecolor("white")
        ax.set_xlim(alim); ax.set_ylim(blim)
        ax.axhline(0, color="#ccc", lw=0.6, zorder=0)
        ax.axvline(0, color="#ccc", lw=0.6, zorder=0)
        ax.set_xlabel(r"$a^*$  (green $\leftarrow$ 0 $\rightarrow$ red)")
        ax.set_ylabel(r"$b^*$  (blue $\leftarrow$ 0 $\rightarrow$ yellow)")
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444")

    # panel 1 — foreground (true pixel colours)
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(a[fg_idx], b[fg_idx], c=rgb_flat[fg_idx],
                s=4, alpha=0.55, linewidths=0, rasterized=True)
    style_ax(ax1)
    ax1.set_title(f"(a) Foreground  ($n={fg_idx.sum():,}$)")

    # panel 2 — background (true pixel colours)
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(a[bg_idx], b[bg_idx], c=rgb_flat[bg_idx],
                s=2, alpha=0.25, linewidths=0, rasterized=True)
    style_ax(ax2)
    ax2.set_title(f"(b) Background  ($n={bg_idx.sum():,}$)")

    # panel 3 — overlay: bg then fg on top
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(a[bg_idx], b[bg_idx], c="#e6850e", s=4, alpha=0.20,
                linewidths=0, rasterized=True, label=f"Background ($n={bg_idx.sum():,}$)")
    ax3.scatter(a[fg_idx], b[fg_idx], c="#31a354", s=4, alpha=0.20,
                linewidths=0, rasterized=True, label=f"Foreground ($n={fg_idx.sum():,}$)")
    style_ax(ax3)
    ax3.legend(loc="upper right", markerscale=2)
    ax3.set_title("(c) Overlay")

    fig.suptitle(r"CIE $a^*b^*$ colour distribution: foreground vs. background",
                 fontsize=11, y=0.97)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.rcParams.update(plt.rcParamsDefault)   # restore defaults
    print(f"[saved] {out_path}")


def main():
    if len(sys.argv) < 3:
        print("usage: python lab_pointcloud.py <image> <labels.json> [out.png]")
        sys.exit(1)

    img_path  = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    out_path  = Path(sys.argv[3]) if len(sys.argv) > 3 \
                else json_path.with_suffix(".lab_plot.png")

    print(f"[1/3] Loading image  {img_path}")
    img_f32 = load_image(str(img_path))

    print(f"[2/3] Loading labels {json_path}")
    data = json.load(open(json_path))
    mask = decode_mask(data["pixel_mask_b64"], data["width"], data["height"])
    print(f"      fg={( mask==1).sum():,}  bg={(mask==0).sum():,}  uncertain={(mask==2).sum():,} px")

    print(f"[3/3] Plotting → {out_path}")
    plot(img_f32, mask, str(out_path))


# ── entry point ───────────────────────────────────────────────────────────
def run():
    IMAGE_PATH  = "wolf.png"
    LABELS_PATH = "segmentation_labels.json"
    OUTPUT_PATH = "wolf_lab_pointcloud.png"

    img_f32 = load_image(IMAGE_PATH)
    data    = json.load(open(LABELS_PATH))
    mask    = decode_mask(data["pixel_mask_b64"], data["width"], data["height"])
    print(f"fg={(mask==1).sum():,}  bg={(mask==0).sum():,}  uncertain={(mask==2).sum():,} px")
    plot(img_f32, mask, OUTPUT_PATH)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main()
    else:
        run()   # hardcoded paths for quick test