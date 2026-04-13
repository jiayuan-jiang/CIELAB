"""
demo/app.py
-----------
Flask server for the segmentation + color/brightness analysis tool.

Routes:
  GET  /             → index.html
  POST /api/slic     → run SLIC on uploaded image, return compact labels
  POST /api/analyze  → compute metrics from pixel mask, return charts + CSV
"""

import uuid
import base64
import io as pyio
import csv as csvmod
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from flask import Flask, request, jsonify, send_file
from skimage import color as skcolor, segmentation as skseg
from PIL import Image

app = Flask(__name__)

# In-memory session store: session_id -> float32 H×W×3 image array
_sessions: dict[str, np.ndarray] = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_image(data: bytes) -> np.ndarray:
    """Image bytes → float32 H×W×3 in [0, 1]."""
    img = Image.open(pyio.BytesIO(data)).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def _fig_to_b64(fig) -> str:
    """Render matplotlib figure to base64 PNG data URL, then close it."""
    buf = pyio.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _cov_ellipse(mean, cov, n_std=2.0, **kw):
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.abs(eigvals))
    return Ellipse(xy=mean, width=w, height=h, angle=theta, **kw)


def _subsample(arr: np.ndarray, n: int = 5000) -> np.ndarray:
    step = max(1, len(arr) // n)
    return arr[::step]


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file(Path(__file__).with_name('index.html'))


@app.route('/api/slic', methods=['POST'])
def api_slic():
    f = request.files.get('image')
    if not f:
        return jsonify({'error': 'No image uploaded.'}), 400

    img_arr = _read_image(f.read())
    H, W = img_arr.shape[:2]

    # SLIC superpixel segmentation
    labels = skseg.slic(img_arr, n_segments=200, compactness=10,
                        sigma=1, start_label=0, channel_axis=2)

    # Encode original image as PNG data URL for the canvas
    pil = Image.fromarray((img_arr * 255).astype(np.uint8))
    img_buf = pyio.BytesIO()
    pil.save(img_buf, format='PNG')
    img_b64 = 'data:image/png;base64,' + base64.b64encode(img_buf.getvalue()).decode()

    # Encode SLIC labels as little-endian uint16 binary (much smaller than pixel lists)
    labels_b64 = base64.b64encode(labels.astype('<u2').tobytes()).decode()

    sid = str(uuid.uuid4())
    _sessions[sid] = img_arr

    return jsonify({
        'session_id': sid,
        'width': int(W),
        'height': int(H),
        'image_b64': img_b64,
        'slic_labels_b64': labels_b64,
        'n_segments': int(labels.max()) + 1,
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    body = request.get_json()
    sid = body.get('session_id', '')
    if sid not in _sessions:
        return jsonify({'error': 'Session expired — please re-upload the image.'}), 404

    img_arr = _sessions[sid]
    H, W = img_arr.shape[:2]

    # Decode pixel mask (0=bg, 1=fg, 2=uncertain)
    mask_bytes = base64.b64decode(body['pixel_mask_b64'])
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(H, W)
    flat = mask.ravel()
    fg = flat == 1
    bg = flat == 0

    if fg.sum() < 10:
        return jsonify({'error': 'Too few foreground pixels — paint more of the subject.'}), 400
    if bg.sum() < 10:
        return jsonify({'error': 'Too few background pixels — leave more regions unlabeled.'}), 400

    # ── CIE a*b* color metrics ────────────────────────────────────────────
    lab = skcolor.rgb2lab(img_arr)
    ab = lab[:, :, 1:3].reshape(-1, 2)

    fg_ab = ab[fg]
    bg_ab = ab[bg]

    mean_fg = np.mean(fg_ab, axis=0)
    mean_bg = np.mean(bg_ab, axis=0)
    dist_centroid = float(np.linalg.norm(mean_fg - mean_bg))

    eps = np.eye(2) * 1e-6
    cov_fg = np.cov(fg_ab, rowvar=False) + eps
    cov_bg = np.cov(bg_ab, rowvar=False) + eps
    cov_pool = (cov_fg + cov_bg) / 2.0
    dm = mean_fg - mean_bg
    t1 = float(0.125 * dm @ np.linalg.inv(cov_pool) @ dm)
    t2 = float(0.5 * np.log(
        np.linalg.det(cov_pool) / np.sqrt(np.linalg.det(cov_fg) * np.linalg.det(cov_bg))
    ))
    dist_bhatt = t1 + t2

    # ── Brightness: luma (no CIE mapping) ────────────────────────────────
    # Y = 0.299R + 0.587G + 0.114B  (ITU-R BT.601, values in [0, 1])
    luma = (0.299 * img_arr[:, :, 0] +
            0.587 * img_arr[:, :, 1] +
            0.114 * img_arr[:, :, 2]).ravel()

    fg_luma = luma[fg]
    bg_luma = luma[bg]
    mean_fg_bright = float(np.mean(fg_luma))
    mean_bg_bright = float(np.mean(bg_luma))
    bright_diff = mean_fg_bright - mean_bg_bright

    # ── Charts ────────────────────────────────────────────────────────────
    rgb_flat = img_arr.reshape(-1, 3)

    c1 = _chart_distribution(fg_ab, bg_ab, rgb_flat, fg, bg)
    c2 = _chart_distances(fg_ab, bg_ab, mean_fg, mean_bg,
                          cov_fg, cov_bg, dist_centroid, dist_bhatt)
    c3 = _chart_brightness(fg_luma, bg_luma, mean_fg_bright, mean_bg_bright)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_rows = [
        ['metric', 'foreground', 'background', 'delta_or_distance'],
        ['centroid_distance_ab', '—', '—', f'{dist_centroid:.4f}'],
        ['bhattacharyya_distance_ab', '—', '—', f'{dist_bhatt:.4f}'],
        ['mean_a_star', f'{mean_fg[0]:.4f}', f'{mean_bg[0]:.4f}', f'{mean_fg[0]-mean_bg[0]:.4f}'],
        ['mean_b_star', f'{mean_fg[1]:.4f}', f'{mean_bg[1]:.4f}', f'{mean_fg[1]-mean_bg[1]:.4f}'],
        ['mean_brightness_luma', f'{mean_fg_bright:.4f}', f'{mean_bg_bright:.4f}', f'{bright_diff:.4f}'],
        ['pixel_count', str(int(fg.sum())), str(int(bg.sum())), '—'],
    ]
    csv_buf = pyio.StringIO()
    csvmod.writer(csv_buf).writerows(csv_rows)

    return jsonify({
        'metrics': {
            'centroid_distance': dist_centroid,
            'bhattacharyya': dist_bhatt,
            'mean_fg_a': float(mean_fg[0]), 'mean_fg_b': float(mean_fg[1]),
            'mean_bg_a': float(mean_bg[0]), 'mean_bg_b': float(mean_bg[1]),
            'mean_fg_brightness': mean_fg_bright,
            'mean_bg_brightness': mean_bg_bright,
            'brightness_diff': bright_diff,
            'fg_pixels': int(fg.sum()),
            'bg_pixels': int(bg.sum()),
        },
        'charts': [c1, c2, c3],
        'chart_titles': [
            'CIE a*b* Color Distribution',
            'Distance Metrics in CIE a*b* Space',
            'Brightness Distribution (Luma)',
        ],
        'csv': csv_buf.getvalue(),
    })


# ── plotting ──────────────────────────────────────────────────────────────────

_RC_BASE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'legend.fontsize': 8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#aaa',
}


def _ab_limits(fg_ab: np.ndarray, bg_ab: np.ndarray, pad: float = 8):
    all_ab = np.concatenate([fg_ab, bg_ab])
    return (
        (float(all_ab[:, 0].min()) - pad, float(all_ab[:, 0].max()) + pad),
        (float(all_ab[:, 1].min()) - pad, float(all_ab[:, 1].max()) + pad),
    )


def _style_ab_ax(ax, alim, blim, title):
    ax.set_facecolor('white')
    ax.set_xlim(alim)
    ax.set_ylim(blim)
    ax.axhline(0, color='#ccc', lw=0.6, zorder=0)
    ax.axvline(0, color='#ccc', lw=0.6, zorder=0)
    ax.set_xlabel(r"$a^*$  (green $\leftarrow$ 0 $\rightarrow$ red)")
    ax.set_ylabel(r"$b^*$  (blue $\leftarrow$ 0 $\rightarrow$ yellow)")
    ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)


def _chart_distribution(fg_ab, bg_ab, rgb_flat, fg_mask, bg_mask) -> str:
    with plt.rc_context({**_RC_BASE}):
        fig = plt.figure(figsize=(13, 4.4), facecolor='white')
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                               left=0.07, right=0.97, top=0.84, bottom=0.14)
        alim, blim = _ab_limits(fg_ab, bg_ab)

        sfg = _subsample(fg_ab)
        sbg = _subsample(bg_ab)
        cfg = _subsample(rgb_flat[fg_mask])
        cbg = _subsample(rgb_flat[bg_mask])

        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(sfg[:, 0], sfg[:, 1], c=cfg, s=4, alpha=0.55,
                    linewidths=0, rasterized=True)
        _style_ab_ax(ax1, alim, blim, f'(a) Foreground   n={len(fg_ab):,}')

        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(sbg[:, 0], sbg[:, 1], c=cbg, s=2, alpha=0.25,
                    linewidths=0, rasterized=True)
        _style_ab_ax(ax2, alim, blim, f'(b) Background   n={len(bg_ab):,}')

        ax3 = fig.add_subplot(gs[2])
        ax3.scatter(sbg[:, 0], sbg[:, 1], c='#e6850e', s=4, alpha=0.20,
                    linewidths=0, rasterized=True, label='Background')
        ax3.scatter(sfg[:, 0], sfg[:, 1], c='#31a354', s=4, alpha=0.20,
                    linewidths=0, rasterized=True, label='Foreground')
        ax3.legend(loc='upper right', markerscale=2)
        _style_ab_ax(ax3, alim, blim, '(c) Overlay')

        fig.suptitle(r"CIE $a^*b^*$ colour distribution — foreground vs. background",
                     fontsize=11, y=0.97)
        return _fig_to_b64(fig)


def _chart_distances(fg_ab, bg_ab, mean_fg, mean_bg,
                     cov_fg, cov_bg, dist_c, dist_b) -> str:
    with plt.rc_context({**_RC_BASE, 'font.size': 10}):
        fig = plt.figure(figsize=(14, 6), facecolor='white')
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

        alim, blim = _ab_limits(fg_ab, bg_ab, pad=12)
        sfg = _subsample(fg_ab)
        sbg = _subsample(bg_ab)
        props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#ccc')

        def fmt(ax, title):
            ax.set_facecolor('white')
            ax.set_xlim(alim); ax.set_ylim(blim)
            ax.axhline(0, color='#ccc', lw=0.8, ls='--', zorder=0)
            ax.axvline(0, color='#ccc', lw=0.8, ls='--', zorder=0)
            ax.set_xlabel(r"$a^*$  (Green–Red)")
            ax.set_ylabel(r"$b^*$  (Blue–Yellow)")
            ax.set_title(title, pad=12, fontweight='bold')
            ax.tick_params(direction='in', top=True, right=True)

        # Panel 1 — centroid distance
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(sbg[:, 0], sbg[:, 1], c='#e6850e', s=2, alpha=0.15, rasterized=True)
        ax1.scatter(sfg[:, 0], sfg[:, 1], c='#31a354', s=2, alpha=0.15, rasterized=True)
        ax1.plot(*mean_bg, marker='X', ms=10, color='#b35f00',
                 markeredgecolor='white', label='BG centroid')
        ax1.plot(*mean_fg, marker='X', ms=10, color='#1e6b36',
                 markeredgecolor='white', label='FG centroid')
        ax1.plot([mean_bg[0], mean_fg[0]], [mean_bg[1], mean_fg[1]],
                 color='black', lw=2, zorder=5)
        ax1.text(0.05, 0.95,
                 "Centroid Euclidean Distance\n\n"
                 r"$D_c = \|\mu_{fg} - \mu_{bg}\|_2$" + f"\n\nResult: {dist_c:.3f}",
                 transform=ax1.transAxes, fontsize=10, va='top', bbox=props)
        ax1.legend(loc='lower right')
        fmt(ax1, '(a) Centroid Distance')

        # Panel 2 — Bhattacharyya
        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(sbg[:, 0], sbg[:, 1], c='#e6850e', s=2, alpha=0.10, rasterized=True)
        ax2.scatter(sfg[:, 0], sfg[:, 1], c='#31a354', s=2, alpha=0.10, rasterized=True)
        ax2.add_patch(_cov_ellipse(mean_bg, cov_bg, edgecolor='#b35f00',
                                   facecolor='none', lw=2, label=r'BG 2$\sigma$'))
        ax2.add_patch(_cov_ellipse(mean_fg, cov_fg, edgecolor='#1e6b36',
                                   facecolor='none', lw=2, label=r'FG 2$\sigma$'))
        ax2.plot(*mean_bg, '+', ms=8, color='#b35f00')
        ax2.plot(*mean_fg, '+', ms=8, color='#1e6b36')
        ax2.text(0.05, 0.95,
                 "Bhattacharyya Distance\n\n"
                 r"$D_B = \frac{1}{8}\Delta\mu^T\Sigma^{-1}\Delta\mu"
                 r"+ \frac{1}{2}\ln\frac{|\Sigma|}{\sqrt{|\Sigma_{fg}||\Sigma_{bg}|}}$"
                 + f"\n\nResult: {dist_b:.3f}",
                 transform=ax2.transAxes, fontsize=10, va='top', bbox=props)
        ax2.legend(loc='lower right')
        fmt(ax2, '(b) Bhattacharyya Distance')

        fig.suptitle(r"Point Cloud Separation in CIE $a^*b^*$ Space",
                     fontsize=14, y=0.98)
        return _fig_to_b64(fig)


def _chart_brightness(fg_luma, bg_luma, mean_fg, mean_bg) -> str:
    with plt.rc_context({**_RC_BASE, 'font.size': 10}):
        fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
        ax.set_facecolor('white')

        bins = np.linspace(0, 1, 64)
        ax.hist(bg_luma, bins=bins, density=True, alpha=0.45,
                color='#e6850e', label=f'Background   μ = {mean_bg:.3f}')
        ax.hist(fg_luma, bins=bins, density=True, alpha=0.45,
                color='#31a354', label=f'Foreground   μ = {mean_fg:.3f}')
        ax.axvline(mean_fg, color='#1e6b36', lw=2, ls='--')
        ax.axvline(mean_bg, color='#b35f00', lw=2, ls='--')

        diff = mean_fg - mean_bg
        ax.set_xlabel(r'Luma   $Y = 0.299\,R + 0.587\,G + 0.114\,B$  (raw, no CIE)')
        ax.set_ylabel('Density')
        ax.set_title(f'Brightness Distribution   FG − BG = {diff:+.4f}',
                     fontweight='bold')
        ax.legend()
        ax.tick_params(direction='in', top=True, right=True)

        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#ccc')
        ax.text(0.98, 0.97, f'ΔY = {diff:+.4f}', transform=ax.transAxes,
                fontsize=12, ha='right', va='top', bbox=props)

        fig.suptitle('Pixel Brightness: Foreground vs. Background', fontsize=12, y=1.0)
        fig.tight_layout()
        return _fig_to_b64(fig)


if __name__ == '__main__':
    app.run(debug=True, port=5050)
