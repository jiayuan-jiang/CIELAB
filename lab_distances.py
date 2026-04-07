"""
lab_distances.py
----------------
Reads segmentation_labels.json + original image.
Calculates and plots distance metrics (Centroid, Bhattacharyya)
between foreground and background point clouds in CIE a*b* space.

Usage:
    CLI: python lab_distances.py <image_path> <labels_json> [output_png]
    IDE: Run directly to use the hardcoded paths in the `run()` function.
"""

import sys
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
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
    raw = base64.b64decode(b64)
    flat = np.frombuffer(raw, dtype=np.uint8)
    return flat.reshape(height, width)

def get_covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
    """Returns a matplotlib Ellipse patch representing the covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Angle of the first principal component
    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Width and height are 2 * n_std * sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(eigvals)
    return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

def plot_distances(img_f32, mask, out_path):
    # Convert to LAB and extract a*b*
    lab = color.rgb2lab(img_f32)
    ab = lab[:, :, 1:3].reshape(-1, 2)
    flat_mask = mask.ravel()

    fg_ab = ab[flat_mask == 1]
    bg_ab = ab[flat_mask == 0]

    # --- Metrics Calculation ---

    # 1. Centroid Distance (Euclidean)
    mean_fg = np.mean(fg_ab, axis=0)
    mean_bg = np.mean(bg_ab, axis=0)
    dist_centroid = np.linalg.norm(mean_fg - mean_bg)

    # 2. Bhattacharyya Distance
    # Add a tiny epsilon to the diagonal to prevent singular matrices
    eps = np.eye(2) * 1e-6
    cov_fg = np.cov(fg_ab, rowvar=False) + eps
    cov_bg = np.cov(bg_ab, rowvar=False) + eps

    cov_pool = (cov_fg + cov_bg) / 2.0
    diff_mean = mean_fg - mean_bg

    # Formula components
    term1 = 0.125 * diff_mean.T @ np.linalg.inv(cov_pool) @ diff_mean
    det_pool = np.linalg.det(cov_pool)
    det_fg = np.linalg.det(cov_fg)
    det_bg = np.linalg.det(cov_bg)
    term2 = 0.5 * np.log(det_pool / np.sqrt(det_fg * det_bg))
    dist_bhatt = term1 + term2

    # --- Plotting Setup ---
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         10,
        "axes.linewidth":    0.8,
        "legend.fontsize":   9,
    })

    fig = plt.figure(figsize=(14, 6), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

    # Shared axis limits based on full distribution
    pad = 10
    alim = (ab[:, 0].min() - pad, ab[:, 0].max() + pad)
    blim = (ab[:, 1].min() - pad, ab[:, 1].max() + pad)

    def format_ax(ax, title):
        ax.set_facecolor("white")
        ax.set_xlim(alim)
        ax.set_ylim(blim)
        ax.axhline(0, color="#ccc", lw=0.8, zorder=0, ls="--")
        ax.axvline(0, color="#ccc", lw=0.8, zorder=0, ls="--")
        ax.set_xlabel(r"$a^*$  (Green-Red Axis)")
        ax.set_ylabel(r"$b^*$  (Blue-Yellow Axis)")
        ax.set_title(title, pad=15, fontweight="bold")
        ax.tick_params(direction="in", top=True, right=True)

    # Subsample for faster scatter plotting without losing visual density
    sub_fg = fg_ab[::max(1, len(fg_ab)//5000)]
    sub_bg = bg_ab[::max(1, len(bg_ab)//5000)]

    # ---------------------------------------------------------
    # Panel 1: Centroid Distance
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(sub_bg[:, 0], sub_bg[:, 1], c="#e6850e", s=2, alpha=0.15, rasterized=True)
    ax1.scatter(sub_fg[:, 0], sub_fg[:, 1], c="#31a354", s=2, alpha=0.15, rasterized=True)

    # Plot Centroids
    ax1.plot(*mean_bg, marker="X", markersize=10, color="#b35f00", markeredgecolor="white", label="BG Centroid")
    ax1.plot(*mean_fg, marker="X", markersize=10, color="#1e6b36", markeredgecolor="white", label="FG Centroid")

    # Draw line connecting them
    ax1.plot([mean_bg[0], mean_fg[0]], [mean_bg[1], mean_fg[1]],
             color="black", linestyle="-", linewidth=2, zorder=5)

    # Centroid Formula Box
    formula_1 = (
        "Metric: Centroid Euclidean Distance\n\n"
        r"$D_c = \sqrt{(a^*_{fg} - a^*_{bg})^2 + (b^*_{fg} - b^*_{bg})^2}$" + "\n\n"
        f"Result: {dist_centroid:.3f}"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#ccc')
    ax1.text(0.05, 0.95, formula_1, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    ax1.legend(loc="lower right")
    format_ax(ax1, "(a) Centroid Distance Diagram")

    # ---------------------------------------------------------
    # Panel 2: Bhattacharyya Distance
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(sub_bg[:, 0], sub_bg[:, 1], c="#e6850e", s=2, alpha=0.10, rasterized=True)
    ax2.scatter(sub_fg[:, 0], sub_fg[:, 1], c="#31a354", s=2, alpha=0.10, rasterized=True)

    # Plot Covariance Ellipses (2 Standard Deviations ~ 95% of data)
    ell_bg = get_covariance_ellipse(mean_bg, cov_bg, n_std=2.0, edgecolor='#b35f00', facecolor='none', lw=2, label="BG Covariance (2$\sigma$)")
    ell_fg = get_covariance_ellipse(mean_fg, cov_fg, n_std=2.0, edgecolor='#1e6b36', facecolor='none', lw=2, label="FG Covariance (2$\sigma$)")
    ax2.add_patch(ell_bg)
    ax2.add_patch(ell_fg)

    # Plot Centroids
    ax2.plot(*mean_bg, marker="+", markersize=8, color="#b35f00")
    ax2.plot(*mean_fg, marker="+", markersize=8, color="#1e6b36")

    # Bhattacharyya Formula Box
    formula_2 = (
        "Metric: Bhattacharyya Distance\n\n"
        r"$D_B = \frac{1}{8}(\mu_{fg}-\mu_{bg})^T \Sigma^{-1} (\mu_{fg}-\mu_{bg}) + \frac{1}{2}\ln\left(\frac{|\Sigma|}{\sqrt{|\Sigma_{fg}||\Sigma_{bg}|}}\right)$" + "\n\n"
        f"Result: {dist_bhatt:.3f}"
    )
    ax2.text(0.05, 0.95, formula_2, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    ax2.legend(loc="lower right")
    format_ax(ax2, "(b) Bhattacharyya Distance Diagram")

    # Save Figure
    fig.suptitle("Point Cloud Separation Metrics in CIE $a^*b^*$ Space", fontsize=14, y=0.98)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Success] Saved metrics visualization to {out_path}")
    print(f"          Centroid Distance:     {dist_centroid:.4f}")
    print(f"          Bhattacharyya Distance: {dist_bhatt:.4f}")

def main():
    """Handles CLI arguments if provided."""
    img_path  = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    out_path  = Path(sys.argv[3]) if len(sys.argv) > 3 \
                else json_path.with_suffix(".metrics.png")

    print(f"[CLI Mode] Loading image {img_path} and labels {json_path}...")
    img_f32 = load_image(str(img_path))
    data = json.load(open(json_path))
    mask = decode_mask(data["pixel_mask_b64"], data["width"], data["height"])

    plot_distances(img_f32, mask, str(out_path))

def run():
    """Fallback for IDE execution or running without arguments."""
    # Modify these paths directly for your local testing
    IMAGE_PATH  = "wolf.png"
    LABELS_PATH = "segmentation_labels.json"
    OUTPUT_PATH = "wolf_distances_metrics.png"

    print(f"[IDE Mode] Using hardcoded paths. Loading {IMAGE_PATH}...")
    try:
        img_f32 = load_image(IMAGE_PATH)
        data = json.load(open(LABELS_PATH))
        mask = decode_mask(data["pixel_mask_b64"], data["width"], data["height"])
        plot_distances(img_f32, mask, OUTPUT_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. Make sure your hardcoded paths are correct.\n{e}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main()
    else:
        run()