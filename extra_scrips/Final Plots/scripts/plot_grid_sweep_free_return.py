from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EASY SETTINGS
# ============================================================

BIG_FONT_SCALE = 1.8

# If your plot shows ~232 deg but you want ~128 deg:
# 232 - 360 = -128, then PHASE_SIGN=-1 gives +128.
PHASE_SIGN = -1

# Zoom limits
LUNAR_X_LIM_DEG = None
LUNAR_Y_LIM_KMS = None

SUCCESS_X_LIM_DEG = (95, 150)
SUCCESS_Y_LIM_KMS = (3.06, 3.16)

CMAP = "viridis"


# ============================================================
# PATH SETUP
# ============================================================

FINAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FINAL_ROOT.parent
STYLE_DIR = FINAL_ROOT / "style"

sys.path.insert(0, str(FINAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STYLE_DIR))

from style.thesis_style import apply_thesis_style, save_thesis_figure


# ============================================================
# STYLE
# ============================================================

def apply_big_heatmap_style():
    apply_thesis_style()

    plt.rcParams.update({
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })


# ============================================================
# DATA LOADING
# ============================================================

def find_npz(data_dir):
    files = list(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz file found in: {data_dir}")
    return files[0]


def load_grid_data(data_dir):
    npz_path = find_npz(data_dir)
    data = np.load(npz_path)

    theta_rad = data["theta"]
    dv_kms = data["dv_kms"]
    moon = data["moon"]
    success = data["success"]

    theta_deg_raw = np.rad2deg(theta_rad)

    # Convert 0...360 deg to -180...180 deg
    theta_deg = (theta_deg_raw + 180.0) % 360.0 - 180.0

    # Optional sign flip so -128 becomes +128
    theta_deg = PHASE_SIGN * theta_deg

    # Sort x-axis after wrapping/sign flip
    order = np.argsort(theta_deg)
    theta_deg = theta_deg[order]
    moon = moon[:, order]
    success = success[:, order]

    print(f"Loaded: {npz_path}")
    print(f"theta range after conversion: {theta_deg.min():.2f} to {theta_deg.max():.2f} deg")
    print(f"dv range: {dv_kms.min():.3f} to {dv_kms.max():.3f} km/s")

    return theta_deg, dv_kms, moon, success


# ============================================================
# PLOTTING
# ============================================================

def plot_heatmap(
    x_deg,
    y_kms,
    z,
    title,
    cbar_label,
    output_path,
    xlim=None,
    ylim=None,
    vmin=None,
    vmax=None,
):
    fig, ax = plt.subplots(
        figsize=(11.5, 6.8),
        constrained_layout=True,
    )

    extent = [
        x_deg.min(),
        x_deg.max(),
        y_kms.min(),
        y_kms.max(),
    ]

    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Phase angle (deg)", labelpad=8)
    ax.set_ylabel(r"$\Delta v$ (km/s)", labelpad=8)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(direction="in", top=True, right=True, pad=6)

    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label(cbar_label, fontsize=24, labelpad=14)
    cbar.ax.tick_params(labelsize=20, pad=6)

    save_thesis_figure(fig, output_path)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main(data_dir=None, output_dir=None):
    apply_big_heatmap_style()

    if data_dir is None:
        data_dir = FINAL_ROOT / "data" / "grid_sweep_free_return"

    if output_dir is None:
        output_dir = FINAL_ROOT / "outputs" / "thesis_ready"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    theta_deg, dv_kms, moon, success = load_grid_data(data_dir)

    plot_heatmap(
        theta_deg,
        dv_kms,
        moon,
        title="Lunar Closest Approach",
        cbar_label="Minimum lunar distance",
        output_path=output_dir / "grid_sweep_lunar_closest_approach",
        xlim=LUNAR_X_LIM_DEG,
        ylim=LUNAR_Y_LIM_KMS,
        vmin=np.nanmin(moon),
        vmax=np.nanmax(moon),
    )

    plot_heatmap(
        theta_deg,
        dv_kms,
        success,
        title="Success Map",
        cbar_label="Success flag",
        output_path=output_dir / "grid_sweep_success_map",
        xlim=SUCCESS_X_LIM_DEG,
        ylim=SUCCESS_Y_LIM_KMS,
        vmin=0,
        vmax=1,
    )

    print("Saved grid sweep plots.")


if __name__ == "__main__":
    main()