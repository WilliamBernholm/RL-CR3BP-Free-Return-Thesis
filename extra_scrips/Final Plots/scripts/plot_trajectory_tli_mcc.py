"""
Final thesis trajectory plotting.

Outputs:
    trajectory_tli_rotating.png/.pdf
    trajectory_tli_inertial.png/.pdf
    trajectory_mcc_overview_rotating.png/.pdf
    trajectory_mcc_overview_inertial.png/.pdf
    mcc_correction_overlay.png/.pdf
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from style.thesis_style import ieee_title




# ============================================================
# PATHS
# ============================================================

FINAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FINAL_ROOT.parent
PROJECT_MODULES = FINAL_ROOT / "project_modules"

sys.path.insert(0, str(FINAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_MODULES))


from style.thesis_style import (
    apply_thesis_style,
    get_figsize,
    clean_axis,
    save_thesis_figure,
)

from config import CR3BPConfig
from cr3bp_env_v4 import (
    rk4_step,
    dist_to_primaries,
    earth_moon_positions,
)


# ============================================================
# FIGURE SIZE CONTROL
# ============================================================

TLI_FIGSIZE_KIND = "double"
MCC_OVERVIEW_FIGSIZE_KIND = "double"
MCC_OVERLAY_FIGSIZE_KIND = "double"


# ============================================================
# GENERAL SETTINGS
# ============================================================


MCC_BLUE_LINEWIDTH = 2.4
MCC_OVERLAY_WIDTH_FACTOR = 0.7
MCC_OVERLAY_WIDTH_DECAY = 0.82

SHOW_TITLE = True
ENABLE_INERTIAL = True
SHOW_GRID = True

ROT_XLIM = (-0.5, 1.5)
ROT_YLIM = (-0.38, 0.38)

INERTIAL_XLIM = (-0.15, 1.15)
INERTIAL_YLIM = (-0.15, 1.0)

COLOR_TRAJ = "tab:blue"
COLOR_BALLISTIC = "darkorange"
COLOR_MOON_TRAIL = "gray"

TRAJ_LINEWIDTH = 1.45
REFERENCE_LINEWIDTH = 1.25
OVERLAY_LINEWIDTH = 1.15
GEOMETRY_LINEWIDTH = 0.9

OVERLAY_ALPHA = 1.0
REFERENCE_ALPHA = 1.0
TRAJ_ALPHA = 1.0


# ============================================================
# TLI SETTINGS
# ============================================================

TLI_USE_BALLISTIC_REF_AS_MAIN = True
TLI_ARROW_VISUAL_LENGTH = 0.080
TLI_ARROW_HEAD_WIDTH = 0.012
TLI_ARROW_HEAD_LENGTH = 0.016


# ============================================================
# MCC SETTINGS
# ============================================================

MCC_MAX_OVERLAYS = 5

MCC_DT = 0.00025
MCC_MAX_STEPS = 80000
MCC_PLOT_EVERY = 5

ESCAPE_RADIUS = 1.50

EARTH_RETURN_RADIUS = 0.05

MCC_ARROW_VISUAL_LENGTH = 0.070
MCC_ARROW_HEAD_WIDTH = 0.010
MCC_ARROW_HEAD_LENGTH = 0.014

MCC_COLORS = [
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:cyan",
    "tab:olive",
]


# ============================================================
# LOAD HELPERS
# ============================================================

def find_first_npz(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in: {folder}")

    return files[0]


def load_archive(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def valid_xy(arr):
    arr = np.asarray(arr)
    return arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] >= 2


def first_available(archive, *keys):
    for key in keys:
        if key in archive:
            return archive[key]
    return None


# ============================================================
# FRAME TRANSFORM
# ============================================================

def rotating_to_inertial(xy_rot, t_hist):
    xy_rot = np.asarray(xy_rot, dtype=float)
    t_hist = np.asarray(t_hist, dtype=float)

    n = min(len(xy_rot), len(t_hist))
    xy_rot = xy_rot[:n]
    t_hist = t_hist[:n]

    c = np.cos(t_hist)
    s = np.sin(t_hist)

    x = c * xy_rot[:, 0] - s * xy_rot[:, 1]
    y = s * xy_rot[:, 0] + c * xy_rot[:, 1]

    return np.column_stack([x, y])


def moon_trail_inertial(cfg, t_hist):
    _, moon_rot = earth_moon_positions(cfg.mu)
    moon_xy_rot = np.repeat(moon_rot.reshape(1, 2), len(t_hist), axis=0)
    return rotating_to_inertial(moon_xy_rot, t_hist)


# ============================================================
# AXIS HELPERS
# ============================================================

def setup_rot_axis(ax, cfg):
    earth, moon = earth_moon_positions(cfg.mu)

    ax.add_patch(Circle(
        earth,
        cfg.r_earth_impact,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=0.8,
        zorder=30,
    ))

    ax.add_patch(Circle(
        moon,
        cfg.r_moon_impact,
        facecolor="gray",
        edgecolor="black",
        linewidth=0.8,
        zorder=30,
    ))

    ax.add_patch(Circle(
        moon,
        cfg.r_moon_flyby,
        fill=False,
        linestyle=":",
        edgecolor="black",
        linewidth=GEOMETRY_LINEWIDTH,
        zorder=20,
    ))

    ax.add_patch(Circle(
        earth,
        cfg.rp_min,
        fill=False,
        linestyle="--",
        edgecolor="black",
        linewidth=GEOMETRY_LINEWIDTH,
        zorder=20,
    ))

    ax.add_patch(Circle(
        earth,
        cfg.rp_max,
        fill=False,
        linestyle="--",
        edgecolor="black",
        linewidth=GEOMETRY_LINEWIDTH,
        zorder=20,
    ))

    ax.set_xlabel(r"$x$ rotating frame [nondim]")
    ax.set_ylabel(r"$y$ rotating frame [nondim]")

    clean_axis(ax, grid=SHOW_GRID)

    ax.set_xlim(*ROT_XLIM)
    ax.set_ylim(*ROT_YLIM)
    ax.set_aspect("equal", adjustable="box")


def setup_inertial_axis(ax, cfg):
    ax.add_patch(Circle(
        (0.0, 0.0),
        cfg.r_earth_impact,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=0.8,
        zorder=30,
    ))

    ax.set_xlabel(r"$x$ Earth-centered inertial [nondim]")
    ax.set_ylabel(r"$y$ Earth-centered inertial [nondim]")

    clean_axis(ax, grid=SHOW_GRID)

    ax.set_xlim(*INERTIAL_XLIM)
    ax.set_ylim(*INERTIAL_YLIM)
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# ARROWS
# ============================================================

def draw_fixed_length_arrow(ax, pos, vec, length, color):
    pos = np.asarray(pos, dtype=float).reshape(2)
    vec = np.asarray(vec, dtype=float).reshape(2)

    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return

    direction = vec / norm
    plot_vec = direction * float(length)

    ax.arrow(
        pos[0],
        pos[1],
        plot_vec[0],
        plot_vec[1],
        head_width=MCC_ARROW_HEAD_WIDTH,
        head_length=MCC_ARROW_HEAD_LENGTH,
        length_includes_head=True,
        color=color,
        linewidth=0.8,
        alpha=1.0,
        zorder=50,
    )


def draw_tli_mean_arrow(ax, archive):
    if (
        "burn_pos_rot" not in archive
        or "burn_dv_vec_rot" not in archive
        or "burn_dv_mag" not in archive
    ):
        return

    pos = np.asarray(archive["burn_pos_rot"], dtype=float)
    dv = np.asarray(archive["burn_dv_vec_rot"], dtype=float)
    mag = np.asarray(archive["burn_dv_mag"], dtype=float)

    valid = np.isfinite(mag) & (mag > 0.0)

    if not np.any(valid):
        return

    pos0 = np.mean(pos[valid], axis=0)
    dv_sum = np.sum(dv[valid], axis=0)

    draw_fixed_length_arrow(
        ax,
        pos0,
        dv_sum,
        TLI_ARROW_VISUAL_LENGTH,
        "black",
    )


def draw_mcc_arrow(ax, pos, dv_vec, color):
    pos = np.asarray(pos, dtype=float)
    dv_vec = np.asarray(dv_vec, dtype=float)

    n = np.linalg.norm(dv_vec)
    if n <= 1e-12:
        return

    direction = dv_vec / n
    v = direction * MCC_ARROW_VISUAL_LENGTH

    # black outline
    ax.arrow(
        pos[0], pos[1],
        v[0], v[1],
        head_width=MCC_ARROW_HEAD_WIDTH * 1.35,
        head_length=MCC_ARROW_HEAD_LENGTH * 1.35,
        length_includes_head=True,
        color="black",
        linewidth=1.2,
        zorder=70,
    )

    # colored arrow on top
    ax.arrow(
        pos[0], pos[1],
        v[0], v[1],
        head_width=MCC_ARROW_HEAD_WIDTH,
        head_length=MCC_ARROW_HEAD_LENGTH,
        length_includes_head=True,
        color=color,
        linewidth=0.8,
        zorder=71,
    )


# ============================================================
# BRANCH PROPAGATION
# ============================================================

def propagate_until_return_region_exit(cfg, state0):
    """
    Propagate a ballistic branch from state0.

    Stop when:
        - branch enters rE < EARTH_RETURN_RADIUS after first leaving it,
        - and then exits rE > EARTH_RETURN_RADIUS again,
        - or Moon impact / escape / max steps occurs.

    This keeps only the relevant return-region encounter and avoids long loops.
    """

    state = np.asarray(state0, dtype=float).copy()
    branch = []

    has_left_return_region = False
    has_entered_return_region_again = False

    for i in range(MCC_MAX_STEPS):
        if i % MCC_PLOT_EVERY == 0:
            branch.append(state.copy())

        rE, rM = dist_to_primaries(cfg.mu, state)

        if rE > EARTH_RETURN_RADIUS:
            has_left_return_region = True

        if has_left_return_region and rE <= EARTH_RETURN_RADIUS:
            has_entered_return_region_again = True

        if has_entered_return_region_again and rE > EARTH_RETURN_RADIUS:
            break

        if rM <= cfg.r_moon_impact:
            break

        if np.linalg.norm(state[:2]) > ESCAPE_RADIUS:
            break

        state = rk4_step(cfg.mu, state, MCC_DT)

    return np.asarray(branch, dtype=float)


# ============================================================
# TLI PLOTS
# ============================================================

def plot_tli(data_dir, output_dir):
    apply_thesis_style()
    cfg = CR3BPConfig()

    path = find_first_npz(data_dir)
    arc = load_archive(path)

    if TLI_USE_BALLISTIC_REF_AS_MAIN and valid_xy(first_available(arc, "ballistic_ref_rot_full")):
        traj = np.asarray(arc["ballistic_ref_rot_full"], dtype=float)
        t_hist = np.asarray(first_available(arc, "ballistic_ref_t_hist", "t_hist"), dtype=float)
    else:
        traj = np.asarray(arc["traj_rot_full"], dtype=float)
        t_hist = np.asarray(arc["t_hist"], dtype=float)

    fig, ax = plt.subplots(figsize=get_figsize(TLI_FIGSIZE_KIND))

    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=COLOR_TRAJ,
        linewidth=TRAJ_LINEWIDTH,
        alpha=TRAJ_ALPHA,
        zorder=10,
    )

    draw_tli_mean_arrow(ax, arc)

    setup_rot_axis(ax, cfg)

    if SHOW_TITLE:
        ax.set_title(ieee_title("PPO-TLI rotating-frame trajectory"))

    save_thesis_figure(fig, Path(output_dir) / "trajectory_tli_rotating")
    plt.close(fig)

    if ENABLE_INERTIAL:
        xy = rotating_to_inertial(traj[:, :2], t_hist)

        fig, ax = plt.subplots(figsize=get_figsize(TLI_FIGSIZE_KIND))

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=COLOR_TRAJ,
            linewidth=TRAJ_LINEWIDTH,
            alpha=TRAJ_ALPHA,
            zorder=10,
        )

        moon_xy = moon_trail_inertial(cfg, t_hist)
        ax.plot(
            moon_xy[:, 0],
            moon_xy[:, 1],
            color=COLOR_MOON_TRAIL,
            linewidth=0.9,
            linestyle=":",
            zorder=5,
        )

        setup_inertial_axis(ax, cfg)

        if SHOW_TITLE:
            ax.set_title(ieee_title("PPO-TLI inertial trajectory"))

        save_thesis_figure(fig, Path(output_dir) / "trajectory_tli_inertial")
        plt.close(fig)

    print(f"[OK] TLI trajectory from {path.name}")


# ============================================================
# MCC OVERVIEW PLOTS
# ============================================================

def plot_mcc_overview(data_dir, output_dir):
    apply_thesis_style()
    cfg = CR3BPConfig()

    path = find_first_npz(data_dir)
    arc = load_archive(path)

    traj = np.asarray(arc["traj_rot_full"], dtype=float)
    t_hist = np.asarray(arc["t_hist"], dtype=float)

    fig, ax = plt.subplots(figsize=get_figsize(MCC_OVERVIEW_FIGSIZE_KIND))

    if "ballistic_ref_rot_full" in arc and valid_xy(arc["ballistic_ref_rot_full"]):
        ref = np.asarray(arc["ballistic_ref_rot_full"], dtype=float)
        ax.plot(
            ref[:, 0],
            ref[:, 1],
            color=COLOR_BALLISTIC,
            linestyle=":",
            linewidth=REFERENCE_LINEWIDTH,
            alpha=REFERENCE_ALPHA,
            zorder=5,
        )

    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=COLOR_TRAJ,
        linewidth=TRAJ_LINEWIDTH,
        alpha=TRAJ_ALPHA,
        zorder=10,
    )

    if "burn_pos_rot" in arc and "burn_dv_vec_rot" in arc:
        for p, dv, c in zip(
            arc["burn_pos_rot"],
            arc["burn_dv_vec_rot"],
            MCC_COLORS * 20,
        ):
            ax.scatter(
                p[0],
                p[1],
                s=25,
                color=c,
                edgecolor="black",
                linewidth=0.4,
                zorder=40,
            )
            draw_mcc_arrow(ax, p, dv, c)

    setup_rot_axis(ax, cfg)

    if SHOW_TITLE:
        ax.set_title(ieee_title("PPO-MCC rotating-frame trajectory"))

    save_thesis_figure(fig, Path(output_dir) / "trajectory_mcc_overview_rotating")
    plt.close(fig)

    if ENABLE_INERTIAL:
        xy = rotating_to_inertial(traj[:, :2], t_hist)

        fig, ax = plt.subplots(figsize=get_figsize(MCC_OVERVIEW_FIGSIZE_KIND))

        if "ballistic_ref_rot_full" in arc and "ballistic_ref_t_hist" in arc:
            ref = np.asarray(arc["ballistic_ref_rot_full"], dtype=float)
            ref_t = np.asarray(arc["ballistic_ref_t_hist"], dtype=float)
            ref_xy = rotating_to_inertial(ref[:, :2], ref_t)

            ax.plot(
                ref_xy[:, 0],
                ref_xy[:, 1],
                color=COLOR_BALLISTIC,
                linestyle=":",
                linewidth=REFERENCE_LINEWIDTH,
                alpha=REFERENCE_ALPHA,
                zorder=5,
            )

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=COLOR_TRAJ,
            linewidth=TRAJ_LINEWIDTH,
            alpha=TRAJ_ALPHA,
            zorder=10,
        )

        moon_xy = moon_trail_inertial(cfg, t_hist)
        ax.plot(
            moon_xy[:, 0],
            moon_xy[:, 1],
            color=COLOR_MOON_TRAIL,
            linewidth=0.9,
            linestyle=":",
            zorder=4,
        )

        setup_inertial_axis(ax, cfg)

        if SHOW_TITLE:
            ax.set_title(ieee_title("PPO-MCC inertial trajectory"))

        save_thesis_figure(fig, Path(output_dir) / "trajectory_mcc_overview_inertial")
        plt.close(fig)

    print(f"[OK] MCC overview from {path.name}")


# ============================================================
# MCC CORRECTION OVERLAY
# ============================================================

def get_mcc_events(arc):
    states_before = np.asarray(arc["step_state_before"], dtype=float)
    dv_step = np.asarray(arc["step_dv_mag"], dtype=float)

    burn_pos = np.asarray(arc["burn_pos_rot"], dtype=float)
    burn_dv = np.asarray(arc["burn_dv_vec_rot"], dtype=float)
    burn_mag = np.asarray(arc["burn_dv_mag"], dtype=float)

    valid_step_indices = np.where(np.isfinite(dv_step) & (dv_step > 0.0))[0]

    n = min(
        len(valid_step_indices),
        len(burn_pos),
        len(burn_dv),
        len(burn_mag),
        MCC_MAX_OVERLAYS,
    )

    events = []

    for i in range(n):
        idx = valid_step_indices[i]

        state_before = states_before[idx].copy()
        dv_vec = burn_dv[i].copy()

        # IMPORTANT:
        # Immediate post-MCC state, not step_state_after.
        state_post = state_before.copy()
        state_post[2:4] += dv_vec

        events.append({
            "state_after": state_post,
            "pos": burn_pos[i],
            "dv": dv_vec,
            "mag": burn_mag[i],
            "color": MCC_COLORS[i % len(MCC_COLORS)],
        })

    return events


def plot_mcc_correction_overlay(data_dir, output_dir):
    """
    MCC correction overlay.

    Orange dotted:
        no-MCC ballistic reference

    Blue:
        final controlled PPO-MCC trajectory

    Colored lines:
        cumulative ballistic continuation after each MCC.
        Each line starts at its own MCC point.

    Plot order:
        orange first
        blue second
        colored branches on top
        later branches thinner and higher z-order
    """

    apply_thesis_style()
    cfg = CR3BPConfig()

    path = find_first_npz(data_dir)
    arc = load_archive(path)

    traj = np.asarray(arc["traj_rot_full"], dtype=float)
    events = get_mcc_events(arc)

    fig, ax = plt.subplots(figsize=get_figsize(MCC_OVERLAY_FIGSIZE_KIND))

    # 1. No-MCC reference
    if "ballistic_ref_rot_full" in arc and valid_xy(arc["ballistic_ref_rot_full"]):
        ref = np.asarray(arc["ballistic_ref_rot_full"], dtype=float)

        ax.plot(
            ref[:, 0],
            ref[:, 1],
            color=COLOR_BALLISTIC,
            linestyle=":",
            linewidth=1.4,
            alpha=1.0,
            zorder=5,
        )

    # --------------------------------------------------------
    # Actual full controlled trajectory
    # Blue is thickest and forms the visual reference.
    # --------------------------------------------------------
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=COLOR_TRAJ,
        linewidth=MCC_BLUE_LINEWIDTH,
        alpha=1.0,
        zorder=20,
    )

    # --------------------------------------------------------
    # Cumulative MCC branches
    # Each branch is thinner than the previous one and plotted above.
    # --------------------------------------------------------
    for i, event in enumerate(events):
        color = event["color"]
        branch = propagate_until_return_region_exit(cfg, event["state_after"])

        linewidth_i = MCC_BLUE_LINEWIDTH * MCC_OVERLAY_WIDTH_FACTOR * (MCC_OVERLAY_WIDTH_DECAY ** i)
        zorder_i = 30 + i

        if valid_xy(branch):
            ax.plot(
                branch[:, 0],
                branch[:, 1],
                color=color,
                linestyle="-",
                linewidth=linewidth_i,
                alpha=1.0,
                zorder=zorder_i,
            )

        p = event["pos"]
        dv = event["dv"]

        ax.scatter(
            p[0],
            p[1],
            s=34,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=60 + i,
        )

        draw_mcc_arrow(ax, p, dv, color)

        p = event["pos"]
        dv = event["dv"]

        ax.scatter(
            p[0],
            p[1],
            s=34,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=50 + i,
        )

        draw_mcc_arrow(ax, p, dv, color)

    setup_rot_axis(ax, cfg)

    if SHOW_TITLE:
        ax.set_title(ieee_title("PPO-MCC correction overlay"))

    ax.set_xlim(*ROT_XLIM)
    ax.set_ylim(*ROT_YLIM)
    ax.set_aspect("equal", adjustable="box")

    save_thesis_figure(fig, Path(output_dir) / "mcc_correction_overlay")
    plt.close(fig)

    print(f"[OK] MCC correction overlay from {path.name}")


# ============================================================
# ENTRY POINTS
# ============================================================

def main_tli(data_dir, output_dir):
    plot_tli(Path(data_dir), Path(output_dir))


def main_mcc(data_dir, output_dir):
    plot_mcc_overview(Path(data_dir), Path(output_dir))
    plot_mcc_correction_overlay(Path(data_dir), Path(output_dir))


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":
    main_tli(
        FINAL_ROOT / "data" / "tli_trajectory",
        FINAL_ROOT / "outputs" / "thesis_ready",
    )

    main_mcc(
        FINAL_ROOT / "data" / "mcc_trajectory",
        FINAL_ROOT / "outputs" / "thesis_ready",
    )