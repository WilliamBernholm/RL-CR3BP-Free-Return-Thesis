
"""
money_shot_plot.py

Interactive 3D trajectory plotter for PPO-A / PPO-B Earth-Moon CR3BP eval archives.

Main features
-------------
1) Three modes:
   - "ppoa"     : plot one PPO-A archive
   - "ppob"     : plot one PPO-B archive
   - "compare"  : plot PPO-A in red, PPO-B in blue, with optional ballistic gap fill

2) Two reference frames:
   - "rotating" : CR3BP rotating frame
   - "inertial" : Earth-centered inertial frame reconstructed from rotating-frame states

3) 3D interactive Matplotlib view:
   - rotate, pan, zoom in the plot window

4) Earth and Moon are plotted as scaled spheres.
   - Earth = blue sphere
   - Moon = gray sphere
   - optional ghost Moon at closest spacecraft lunar approach
   - optional Moon trail in inertial frame

5) Optional impulse burn arrows:
   - independent scale factors for PPO-A and PPO-B

6) Optional ballistic overlays:
   - PPO-A: saved ballistic_ref_rot_full if available
   - PPO-B: saved ballistic_ref_rot_full if available
   - compare mode: optional ballistic propagation gap from PPO-A final/TLI state to PPO-B start

Place this file in the same directory as:
- config.py
- cr3bp_env_v4.py
- your "Saved Policies" folder

Run:
    python money_shot_plot.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import numpy as np

# Use an interactive GUI backend. If this fails on your system, comment it out.
import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 1200
plt.rcParams["figure.dpi"] = 160

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    pass

from config import RUN
from cr3bp_env_v4 import (
    earth_moon_positions,
    rk4_step,
    minutes_to_nondim_time,
)


# ============================================================
# USER SETTINGS
# ============================================================


TRAJECTORY_FRACTION = 0.87

# ---------- mode ----------
MODE = "ppob"  # "ppoa", "ppob", "compare"

# ---------- reference frame ----------
FRAME = "inertial"  # "rotating" or "inertial"

# ---------- file selection ----------
SAVED_ROOT = Path(__file__).resolve().parent / RUN.saved_root_name  # usually "Saved Policies"
MANUAL_SELECT = True

# If MANUAL_SELECT = False, set these paths directly.
PPOA_EPISODE_REPORT_JSON = Path("path/to/ppoa_episode_report.json")
PPOB_EPISODE_REPORT_JSON = Path("path/to/ppob_episode_report.json")

# ---------- interactive plot ----------
FIGSIZE = (12, 9)
INITIAL_ELEV = 28
INITIAL_AZIM = -58
INITIAL_DIST = 9  # may be ignored by newer matplotlib versions

FULLSCREEN_WINDOW = True
FILL_FIGURE_CANVAS = True
TITLE_PAD = 2

CLIP_TRAJECTORIES_ON_IMPACT = True
IMPACT_PADDING_ND = 0.0  # increase slightly, e.g. 0.001, if you want earlier cutoff

# ---------- title / font / axes ----------
FIG_TITLE = ""
TITLE_SIZE = 18
LABEL_SIZE = 13
TICK_SIZE = 10
LEGEND_SIZE = 10

SHOW_GRID = False
SHOW_LEGEND = False
SHOW_AXIS_LABELS = True
SHOW_AXIS_TICKS = True
EQUAL_AXIS_SCALE = True
SHOW_AXES = False  # False = hide all axes, ticks, labels, panes
SHOW_GHOST_MOON_TEXT = False



# ---------- trajectory styling ----------
PPOA_COLOR = "red"
PPOB_COLOR = "#1B2A7A"
PPOA_LINEWIDTH = 2.3
PPOB_LINEWIDTH = 2.3

GAP_COLOR = "red"
GAP_LINEWIDTH = 2.0
GAP_LINESTYLE = "--"

BALLISTIC_PPOA_COLOR = "darkred"
BALLISTIC_PPOB_COLOR = "navy"
BALLISTIC_LINEWIDTH = 1.6
BALLISTIC_LINESTYLE = ":"

MOON_TRAIL_COLOR = "gray"
MOON_TRAIL_LINEWIDTH = 1.0
MOON_TRAIL_ALPHA = 0.45

CLIP_TRAJECTORIES_TO_RADIUS = True
CLIP_RADIUS_ND = 1.3

# ---------- object spheres ----------
EARTH_RADIUS_ND = 0.0140
MOON_RADIUS_ND = 0.0045
EARTH_COLOR = "dodgerblue"
MOON_COLOR = "gray"
EARTH_ALPHA = 0.85
MOON_ALPHA = 0.85
SPHERE_RES_U = 36
SPHERE_RES_V = 18

# ---------- ghost Moon at closest approach ----------
SHOW_GHOST_MOON_AT_CLOSEST_APPROACH = True
GHOST_MOON_ALPHA = 0.25
GHOST_MOON_COLOR = "gray"

# ---------- impulse arrows ----------
SHOW_DV_ARROWS = False
DV_ARROW_LENGTH_SCALE_PPOA = 2.0
DV_ARROW_LENGTH_SCALE_PPOB = 10.0
DV_ARROW_LINEWIDTH = 4.0
DV_ARROW_NORMALIZE = False  # False means arrow length still reflects relative dv magnitude
DV_ARROW_PPOA_COLOR = "red"
DV_ARROW_PPOB_COLOR = "red"
DV_ARROW_HEAD_SIZE = 0.08
DV_ARROW_HEAD_WIDTH = 0.04

DV_ARROW_MIN_LENGTH = 0.165
DV_ARROW_MAX_LENGTH = 0.16
DV_ARROW_USE_VISUAL_LENGTH_LIMITS = True


DV_ARROW_OUTLINE_WIDTH = 7.5
DV_ARROW_MAIN_WIDTH = 4.0


USE_3D_CONE_ARROWS = False
DV_ARROW_COLOR = "red"
DV_ARROW_OUTLINE_COLOR = "black"
DV_ARROW_OUTLINE_SCALE = 1.55

DV_ARROW_SHAFT_RADIUS = 0.0055
DV_ARROW_HEAD_RADIUS = 0.020
DV_ARROW_HEAD_LENGTH_FRAC = 0.32

DV_ARROW_ALPHA = 1.0
DV_ARROW_RESOLUTION = 18

# ---------- ballistic overlays ----------
SHOW_BALLISTIC = True
BALLISTIC_MAX_POINTS = 5000

# ---------- PPO-A -> PPO-B gap fill ----------
# This is the important part for your current PPO-A/PPO-B pipeline:
# PPO-A performs TLI, then PPO-B begins after a delay, for example 30 minutes.
# This ballistic segment fills that visual gap.
FILL_PPOA_TO_PPOB_GAP_IN_COMPARE = True

# Propagate PPO-A final state ballistically for this many minutes.
# Set to 30.0 for your current PPO-B handoff assumption.
PPOA_TO_PPOB_GAP_MINUTES = 30.0

# Extra propagation after the nominal gap, useful if you want a slight overlap.
PPOA_TO_PPOB_EXTRA_GAP_MINUTES = 0.0

# If True, start gap from last PPO-A state. Usually correct if PPO-A archive ends after TLI.
# If False, start from the last burn position/state if available. Last PPO-A state is safer.
GAP_START_FROM_PPOA_FINAL_STATE = True

# Visual-only option: translate PPO-B path so its first point equals gap end.
# Use only if the PPO-A archive and PPO-B archive are from the same physical case but
# have tiny numerical mismatch. Leave False for physically honest plotting.
VISUALLY_SNAP_PPOB_START_TO_GAP_END = False

# Print diagnostics about gap mismatch.
PRINT_GAP_DIAGNOSTICS = True

# ---------- inertial frame options ----------
# In inertial mode, the plot is Earth-centered.
# The Moon moves, so we add a Moon trail over the same time history.
SHOW_MOON_TRAIL_IN_INERTIAL = True
MOON_TRAIL_USE_COMBINED_TIME = True

# ---------- output ----------
SAVE_FIGURE = False
OUTPUT_DPI = 1200

# "regular", "dark", or "transparent"
PLOT_THEME = "regular"

# Automatic tagged filenames
AUTO_NAME_OUTPUT = True
ADD_TIMESTAMP_TO_OUTPUT = False

OUTPUT_PATH = Path("money_shot_plot.png")


# ============================================================
# BASIC HELPERS
# ============================================================

def _as_path(x) -> Path:
    return x if isinstance(x, Path) else Path(str(x))


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _archive_paths_from_episode_report(report_path: Path) -> Tuple[Path, Path, Path]:
    """
    Given:
        eval0105_step000430080_episode_report.json

    infer:
        eval0105_step000430080_arrays.npz
        eval0105_step000430080_meta.json
    """
    name = report_path.name
    prefix = name.replace("_episode_report.json", "")
    arrays_path = report_path.with_name(prefix + "_arrays.npz")
    meta_path = report_path.with_name(prefix + "_meta.json")
    return report_path, arrays_path, meta_path


def _find_episode_reports(root: Path) -> List[Path]:
    return sorted(root.rglob("*_episode_report.json"))


def _choose_from_list(items: List[Path], prompt: str) -> Path:
    if not items:
        raise FileNotFoundError(f"No selectable items found for: {prompt}")

    print("\n" + "=" * 90)
    print(prompt)
    print("=" * 90)
    for i, item in enumerate(items):
        print(f"[{i:03d}] {item}")

    while True:
        raw = input("\nSelect index: ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(items):
                return items[idx]
        except Exception:
            pass
        print("Invalid selection. Try again.")


def select_episode_report(label: str) -> Path:
    """
    Simple folder selection:
    Saved Policies -> run -> archive file.
    """
    root = _as_path(SAVED_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"Saved root not found: {root.resolve()}")

    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    run = _choose_from_list(runs, f"{label}: select run inside {root}")

    reports = _find_episode_reports(run)
    report = _choose_from_list(reports, f"{label}: select eval episode report")

    return report


def load_archive(report_path: Path) -> Dict[str, Any]:
    report_path, arrays_path, meta_path = _archive_paths_from_episode_report(report_path)

    if not report_path.exists():
        raise FileNotFoundError(f"Episode report not found: {report_path}")
    if not arrays_path.exists():
        raise FileNotFoundError(f"Arrays file not found: {arrays_path}")

    episode_report = _load_json(report_path)
    meta = _load_json(meta_path) if meta_path.exists() else {}
    arrays = np.load(arrays_path, allow_pickle=True)

    return {
        "report_path": report_path,
        "arrays_path": arrays_path,
        "meta_path": meta_path,
        "episode_report": episode_report,
        "meta": meta,
        "arrays": arrays,
    }


def get_mu_from_archive_or_default(archive: Optional[Dict[str, Any]] = None) -> float:
    """
    Prefer importing config value from Earth/Moon positions. Most project configs use Earth-Moon CR3BP mu.
    """
    # earth_moon_positions only needs mu, but mu is not passed here. In your environment
    # the standard nondim mass parameter is approximately Moon / (Earth + Moon).
    return 0.012150585609624


def get_rotating_primaries(mu: float) -> Tuple[np.ndarray, np.ndarray]:
    rE, rM = earth_moon_positions(mu)
    return np.asarray(rE, dtype=float), np.asarray(rM, dtype=float)


def ensure_state4(traj: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 4), dtype=float)

    if arr.shape[1] >= 4:
        return arr[:, :4]

    out = np.zeros((arr.shape[0], 4), dtype=float)
    out[:, :2] = arr[:, :2]
    return out


def get_traj_and_time(archive: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    arrays = archive["arrays"]
    if "traj_rot_full" in arrays:
        traj = ensure_state4(arrays["traj_rot_full"])
    elif "traj_rot" in arrays:
        traj = ensure_state4(arrays["traj_rot"])
    else:
        raise KeyError("Could not find traj_rot_full or traj_rot in arrays npz.")

    if "t_hist" in arrays:
        t_hist = np.asarray(arrays["t_hist"], dtype=float)
    else:
        t_hist = np.linspace(0.0, 1.0, len(traj))

    n = min(len(traj), len(t_hist))
    return traj[:n], t_hist[:n]


def get_ballistic_ref_and_time(archive: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    arrays = archive["arrays"]
    if "ballistic_ref_rot_full" not in arrays:
        return np.zeros((0, 4)), np.zeros((0,))

    traj = ensure_state4(arrays["ballistic_ref_rot_full"])
    if "ballistic_ref_t_hist" in arrays:
        t_hist = np.asarray(arrays["ballistic_ref_t_hist"], dtype=float)
    else:
        t_hist = np.linspace(0.0, 1.0, len(traj))

    n = min(len(traj), len(t_hist))
    traj = traj[:n]
    t_hist = t_hist[:n]

    if len(traj) > BALLISTIC_MAX_POINTS:
        idx = np.linspace(0, len(traj) - 1, BALLISTIC_MAX_POINTS).astype(int)
        traj = traj[idx]
        t_hist = t_hist[idx]

    return traj, t_hist




def clip_xyz_to_radius(xyz, r_max):
    xyz = np.asarray(xyz, dtype=float)
    if xyz is None or len(xyz) == 0:
        return xyz

    r = np.linalg.norm(xyz[:, :2], axis=1)
    keep = r <= float(r_max)

    if not np.any(keep):
        return np.zeros((0, 3), dtype=float)

    # Keep only the first continuous valid segment
    first_bad = np.where(~keep)[0]
    if len(first_bad) > 0:
        cut = first_bad[0]
        return xyz[:cut]

    return xyz


def clip_xyz_on_impact(xyz, t_hist, frame, mu):
    xyz = np.asarray(xyz, dtype=float)
    t_hist = np.asarray(t_hist, dtype=float)

    if xyz is None or len(xyz) == 0:
        return xyz

    # Safety: make trajectory and time arrays the same length.
    n = min(len(xyz), len(t_hist))
    xyz = xyz[:n]
    t_hist = t_hist[:n]

    earth_xy = np.asarray([
        earth_position_in_plot_frame(frame, mu)[:2]
        for _ in range(n)
    ])

    if frame == "rotating":
        moon_xy = moon_position_in_plot_frame(np.zeros(n), frame, mu)
    else:
        moon_xy = moon_position_in_plot_frame(t_hist, frame, mu)

    rE = np.linalg.norm(xyz[:, :2] - earth_xy, axis=1)
    rM = np.linalg.norm(xyz[:, :2] - moon_xy, axis=1)

    impact = (
        (rE <= EARTH_RADIUS_ND + IMPACT_PADDING_ND) |
        (rM <= MOON_RADIUS_ND + IMPACT_PADDING_ND)
    )

    if np.any(impact):
        cut = int(np.argmax(impact)) + 1
        return xyz[:cut]

    return xyz


def get_burn_arrays(archive: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = archive["arrays"]

    if "burn_pos_rot" in arrays and "burn_dv_vec_rot" in arrays:
        pos = np.asarray(arrays["burn_pos_rot"], dtype=float)
        dv = np.asarray(arrays["burn_dv_vec_rot"], dtype=float)
        mag = np.asarray(arrays["burn_dv_mag"], dtype=float) if "burn_dv_mag" in arrays else np.linalg.norm(dv, axis=1)
        return pos, dv, mag

    # Fallback to meta burn_events.
    burns = archive.get("meta", {}).get("burn_events", [])
    pos = []
    dv = []
    mag = []
    for b in burns:
        if "pos_rot" in b and "dv_vec_rot" in b:
            pos.append(b["pos_rot"])
            dv.append(b["dv_vec_rot"])
            mag.append(b.get("dv_mag", np.linalg.norm(b["dv_vec_rot"])))
    return np.asarray(pos, dtype=float), np.asarray(dv, dtype=float), np.asarray(mag, dtype=float)


# ============================================================
# FRAME TRANSFORMS
# ============================================================

def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotating_to_earth_centered_inertial_xy(
    xy_rot: np.ndarray,
    t_hist: np.ndarray,
    mu: float,
) -> np.ndarray:
    """
    Convert rotating-frame barycentric positions to Earth-centered inertial positions.

    CR3BP rotating frame has angular rate 1 nondim.
    Earth is fixed at rE_rot. For an Earth-centered inertial view:

        r_rel_rot = r_sc_rot - r_E_rot
        r_rel_inertial = R(t) r_rel_rot

    This is enough for plotting geometry.
    """
    xy_rot = np.asarray(xy_rot, dtype=float)
    t_hist = np.asarray(t_hist, dtype=float)

    rE_rot, _ = get_rotating_primaries(mu)
    out = np.zeros_like(xy_rot[:, :2])

    for i in range(len(out)):
        out[i] = rot2(t_hist[i]) @ (xy_rot[i, :2] - rE_rot)

    return out


def rotating_vector_to_inertial_xy(
    vec_rot: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    """
    Rotate a vector from rotating axes to inertial axes.
    This is used for dv arrows.
    """
    vec_rot = np.asarray(vec_rot, dtype=float)
    t_values = np.asarray(t_values, dtype=float)
    out = np.zeros_like(vec_rot[:, :2])
    for i in range(len(out)):
        out[i] = rot2(t_values[i]) @ vec_rot[i, :2]
    return out


def moon_position_in_plot_frame(
    t_values: np.ndarray,
    frame: str,
    mu: float,
) -> np.ndarray:
    """
    Return Moon position in either rotating or Earth-centered inertial plotting frame.
    """
    t_values = np.asarray(t_values, dtype=float)
    rE_rot, rM_rot = get_rotating_primaries(mu)

    if frame == "rotating":
        return np.repeat(np.array([[rM_rot[0], rM_rot[1]]], dtype=float), len(t_values), axis=0)

    moon_rel_rot = rM_rot - rE_rot
    out = np.zeros((len(t_values), 2), dtype=float)
    for i, t in enumerate(t_values):
        out[i] = rot2(t) @ moon_rel_rot
    return out


def earth_position_in_plot_frame(frame: str, mu: float) -> np.ndarray:
    if frame == "rotating":
        rE_rot, _ = get_rotating_primaries(mu)
        return np.array([rE_rot[0], rE_rot[1], 0.0])
    return np.array([0.0, 0.0, 0.0])


def convert_path_to_plot_frame(
    traj_rot: np.ndarray,
    t_hist: np.ndarray,
    frame: str,
    mu: float,
) -> np.ndarray:
    traj_rot = ensure_state4(traj_rot)
    if len(traj_rot) == 0:
        return np.zeros((0, 3), dtype=float)

    if frame == "rotating":
        return np.column_stack([traj_rot[:, 0], traj_rot[:, 1], np.zeros(len(traj_rot))])

    xy = rotating_to_earth_centered_inertial_xy(traj_rot[:, :2], t_hist, mu)
    return np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])


def convert_points_to_plot_frame(
    points_rot: np.ndarray,
    t_values: np.ndarray,
    frame: str,
    mu: float,
) -> np.ndarray:
    points_rot = np.asarray(points_rot, dtype=float)
    if points_rot.size == 0:
        return np.zeros((0, 3), dtype=float)

    if frame == "rotating":
        return np.column_stack([points_rot[:, 0], points_rot[:, 1], np.zeros(len(points_rot))])

    xy = rotating_to_earth_centered_inertial_xy(points_rot[:, :2], t_values, mu)
    return np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])


# ============================================================
# GAP PROPAGATION
# ============================================================

def propagate_ballistic_gap_from_ppoa(
    ppoa_archive: Dict[str, Any],
    gap_minutes: float,
    extra_minutes: float,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate from PPO-A final state using CR3BP RK4 dynamics.

    This fills the visual/physical gap between PPO-A TLI completion and PPO-B start
    when PPO-B starts a fixed time after TLI, e.g. 30 minutes.
    """
    traj_a, t_a = get_traj_and_time(ppoa_archive)
    if len(traj_a) == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    state = np.asarray(traj_a[-1], dtype=float).copy()
    t0 = float(t_a[-1])

    total_minutes = float(gap_minutes) + float(extra_minutes)
    if total_minutes <= 0.0:
        return np.asarray([state], dtype=float), np.asarray([t0], dtype=float)

    # Use a modest substep for smooth visual propagation.
    dt_minutes = min(2.0, max(0.1, total_minutes / 200.0))
    dt = minutes_to_nondim_time(dt_minutes)
    t_end = t0 + minutes_to_nondim_time(total_minutes)
    n_steps = int(math.ceil((t_end - t0) / max(dt, 1e-12)))

    states = [state.copy()]
    times = [t0]
    t = t0

    for _ in range(n_steps):
        step = min(dt, t_end - t)
        if step <= 0:
            break
        state = rk4_step(mu, state, step)
        t += step
        states.append(state.copy())
        times.append(t)

    return np.asarray(states, dtype=float), np.asarray(times, dtype=float)


# ============================================================
# PLOTTING
# ============================================================

def draw_sphere(ax, center_xyz, radius, color, alpha):
    u = np.linspace(0, 2 * np.pi, SPHERE_RES_U)
    v = np.linspace(0, np.pi, SPHERE_RES_V)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center_xyz[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center_xyz[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center_xyz[2]

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def estimate_burn_times_from_nearest_traj(
    burn_pos_rot: np.ndarray,
    traj_rot: np.ndarray,
    t_hist: np.ndarray,
) -> np.ndarray:
    if len(burn_pos_rot) == 0:
        return np.zeros((0,), dtype=float)

    out = np.zeros((len(burn_pos_rot),), dtype=float)
    traj_xy = traj_rot[:, :2]
    for i, p in enumerate(burn_pos_rot):
        d = np.linalg.norm(traj_xy - p[:2], axis=1)
        idx = int(np.argmin(d))
        out[i] = t_hist[idx]
    return out


def _orthonormal_basis_from_direction(direction):
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-12)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(direction, ref)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(direction, e1)
    return direction, e1, e2


def draw_3d_arrow(ax, start, vector, color, outline=False):
    start = np.asarray(start, dtype=float)
    vector = np.asarray(vector, dtype=float)

    length = np.linalg.norm(vector)
    if length < 1e-12:
        return

    direction, e1, e2 = _orthonormal_basis_from_direction(vector)

    scale = DV_ARROW_OUTLINE_SCALE if outline else 1.0
    shaft_radius = DV_ARROW_SHAFT_RADIUS * scale
    head_radius = DV_ARROW_HEAD_RADIUS * scale

    head_length = min(
        length * DV_ARROW_HEAD_LENGTH_FRAC,
        length * 0.75
    )

    shaft_length = max(length - head_length, length * 0.25)

    # Shaft as tube
    theta = np.linspace(0, 2*np.pi, DV_ARROW_RESOLUTION)
    s = np.linspace(0, shaft_length, 2)

    T, S = np.meshgrid(theta, s)

    center = start[None, None, :] + S[:, :, None] * direction[None, None, :]
    circle = (
        np.cos(T)[:, :, None] * e1[None, None, :] +
        np.sin(T)[:, :, None] * e2[None, None, :]
    )

    tube = center + shaft_radius * circle

    ax.plot_surface(
        tube[:, :, 0], tube[:, :, 1], tube[:, :, 2],
        color=color,
        alpha=DV_ARROW_ALPHA,
        linewidth=0,
        shade=True
    )

    # Head as cone
    cone_s = np.linspace(0, head_length, 2)
    T, S = np.meshgrid(theta, cone_s)

    base = start + shaft_length * direction
    radius = head_radius * (1.0 - S / max(head_length, 1e-12))

    center = base[None, None, :] + S[:, :, None] * direction[None, None, :]
    cone = center + radius[:, :, None] * circle[:2, :, :]

    ax.plot_surface(
        cone[:, :, 0], cone[:, :, 1], cone[:, :, 2],
        color=color,
        alpha=DV_ARROW_ALPHA,
        linewidth=0,
        shade=True
    )


def plot_dv_arrows(
    ax,
    archive: Dict[str, Any],
    color: str,
    scale: float,
    frame: str,
    mu: float,
):
    burn_pos_rot, burn_dv_rot, burn_mag = get_burn_arrays(archive)
    if len(burn_pos_rot) == 0:
        return

    traj_rot, t_hist = get_traj_and_time(archive)
    burn_times = estimate_burn_times_from_nearest_traj(burn_pos_rot, traj_rot, t_hist)

    p_plot = convert_points_to_plot_frame(burn_pos_rot, burn_times, frame, mu)

    if frame == "rotating":
        dv_plot_xy = burn_dv_rot[:, :2]
    else:
        dv_plot_xy = rotating_vector_to_inertial_xy(burn_dv_rot[:, :2], burn_times)

    for p, dv in zip(p_plot, dv_plot_xy):
        v = np.array([dv[0], dv[1], 0.0], dtype=float)
        if DV_ARROW_NORMALIZE:
            n = np.linalg.norm(v)
            if n > 1e-12:
                v = v / n
        v = scale * v

        if DV_ARROW_USE_VISUAL_LENGTH_LIMITS:
            length = np.linalg.norm(v)

            if length > 1e-12:
                direction = v / length

                # Enforce minimum and maximum visual arrow length
                visual_length = np.clip(
                    length,
                    DV_ARROW_MIN_LENGTH,
                    DV_ARROW_MAX_LENGTH
                )

                v = direction * visual_length

        if USE_3D_CONE_ARROWS:
            draw_3d_arrow(ax, p, v, DV_ARROW_OUTLINE_COLOR, outline=True)
            draw_3d_arrow(ax, p, v, color, outline=False)
        else:

            ax.quiver(
                p[0], p[1], p[2],
                v[0], v[1], 0.0,
                color=color,
                linewidth=2.5,
                arrow_length_ratio=0.35,
                normalize=False,
            )


def truncate_xyz_fraction(xyz, t_hist=None):
    """
    Keep only the first fraction of the trajectory.
    """

    xyz = np.asarray(xyz)

    if len(xyz) == 0:
        if t_hist is None:
            return xyz
        return xyz, t_hist

    frac = float(TRAJECTORY_FRACTION)

    frac = max(0.0, min(1.0, frac))

    n_keep = max(2, int(frac * len(xyz)))

    xyz_cut = xyz[:n_keep]

    if t_hist is None:
        return xyz_cut

    return xyz_cut, t_hist[:n_keep]


def plot_path(
    ax,
    xyz: np.ndarray,
    color: str,
    linewidth: float,
    label: str,
    linestyle: str = "-",
    t_hist: np.ndarray = None,
    frame: str = None,
    mu: float = None,
):
    if xyz is None or len(xyz) == 0:
        return
    
    xyz, t_hist = truncate_xyz_fraction(xyz, t_hist)

    if CLIP_TRAJECTORIES_ON_IMPACT and t_hist is not None and frame is not None and mu is not None:
        xyz = clip_xyz_on_impact(xyz, t_hist, frame, mu)

    if CLIP_TRAJECTORIES_TO_RADIUS:
        xyz = clip_xyz_to_radius(xyz, CLIP_RADIUS_ND)

    if xyz is None or len(xyz) == 0:
        return

    ax.plot(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


def find_closest_moon_event(
    traj_rot: np.ndarray,
    t_hist: np.ndarray,
    frame: str,
    mu: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Return moon position at closest spacecraft approach to the Moon.
    """
    if len(traj_rot) == 0:
        return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

    _, rM_rot = get_rotating_primaries(mu)
    d = np.linalg.norm(traj_rot[:, :2] - rM_rot, axis=1)
    idx = int(np.argmin(d))
    t_close = float(t_hist[idx])
    moon_xy = moon_position_in_plot_frame(np.asarray([t_close]), frame, mu)[0]
    return np.array([moon_xy[0], moon_xy[1], 0.0]), t_close, float(d[idx])


def set_axes_equal(ax):
    """
    Make 3D axes have equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-6])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def apply_visual_settings(ax):
    ax.set_title(FIG_TITLE, fontsize=TITLE_SIZE, pad=TITLE_PAD)

    if SHOW_AXIS_LABELS:
        if FRAME == "rotating":
            ax.set_xlabel("x rotating [nondim]", fontsize=LABEL_SIZE)
            ax.set_ylabel("y rotating [nondim]", fontsize=LABEL_SIZE)
        else:
            ax.set_xlabel("x Earth-centered inertial [nondim]", fontsize=LABEL_SIZE)
            ax.set_ylabel("y Earth-centered inertial [nondim]", fontsize=LABEL_SIZE)
        ax.set_zlabel("z [visual only]", fontsize=LABEL_SIZE)

    if not SHOW_AXIS_TICKS:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(SHOW_GRID)

    ax.view_init(elev=INITIAL_ELEV, azim=INITIAL_AZIM)
    try:
        ax.dist = INITIAL_DIST
    except Exception:
        pass

    if EQUAL_AXIS_SCALE:
        set_axes_equal(ax)

    if not SHOW_AXES:
        ax.set_axis_off()

    if SHOW_LEGEND:
        ax.legend(fontsize=LEGEND_SIZE)


def plot_moon_trail(ax, all_times: np.ndarray, frame: str, mu: float):
    if frame != "inertial":
        return
    if not SHOW_MOON_TRAIL_IN_INERTIAL:
        return
    if all_times is None or len(all_times) == 0:
        return

    t_min = float(np.nanmin(all_times))
    t_max = float(np.nanmax(all_times))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return

    t_line = np.linspace(t_min, t_max, 600)
    moon_xy = moon_position_in_plot_frame(t_line, frame, mu)
    xyz = np.column_stack([moon_xy[:, 0], moon_xy[:, 1], np.zeros(len(moon_xy))])
    ax.plot(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        color=MOON_TRAIL_COLOR,
        linewidth=MOON_TRAIL_LINEWIDTH,
        alpha=MOON_TRAIL_ALPHA,
        label="Moon trail",
    )


def maybe_shift_archive_for_visual_snap(
    archive: Dict[str, Any],
    shift_rot_xy: np.ndarray,
) -> Dict[str, Any]:
    """
    Visual-only translation of PPO-B arrays.
    This modifies a lightweight copy, not the files.
    """
    if not VISUALLY_SNAP_PPOB_START_TO_GAP_END:
        return archive

    # We cannot safely mutate np.load object directly, so make a wrapper with copied key arrays.
    arrays = archive["arrays"]
    copied = {}
    for k in arrays.keys():
        copied[k] = np.array(arrays[k], copy=True)

    for key in ["traj_rot_full", "ballistic_ref_rot_full", "burn_pos_rot"]:
        if key in copied and copied[key].ndim >= 2 and copied[key].shape[1] >= 2:
            copied[key][:, 0] += shift_rot_xy[0]
            copied[key][:, 1] += shift_rot_xy[1]

    new_archive = dict(archive)
    new_archive["arrays"] = copied
    return new_archive


def make_output_path() -> Path:
    """
    Build a tagged output filename automatically.
    Example:
        moneyshot_compare_inertial_transparent.png
    """
    if not AUTO_NAME_OUTPUT:
        return OUTPUT_PATH

    name = f"moneyshot_{MODE}_{FRAME}_{PLOT_THEME}"

    if ADD_TIMESTAMP_TO_OUTPUT:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name += f"_{timestamp}"

    return Path(name + ".png")


def apply_plot_theme(fig, ax):
    """
    Apply regular, dark, or transparent plot styling.
    Transparent keeps the regular trajectory colors.
    """
    global DV_ARROW_OUTLINE_COLOR

    if PLOT_THEME == "regular":
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        DV_ARROW_OUTLINE_COLOR = "black"

    elif PLOT_THEME == "dark":
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        DV_ARROW_OUTLINE_COLOR = "white"

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.tick_params(colors="white")

    elif PLOT_THEME == "transparent":
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((1, 1, 1, 0))
        DV_ARROW_OUTLINE_COLOR = "white"

    else:
        raise ValueError("PLOT_THEME must be 'regular', 'dark', or 'transparent'.")


def make_money_plot(
    ppoa_archive: Optional[Dict[str, Any]],
    ppob_archive: Optional[Dict[str, Any]],
):
    frame = FRAME.lower().strip()
    if frame not in ("rotating", "inertial"):
        raise ValueError("FRAME must be 'rotating' or 'inertial'.")

    mu = get_mu_from_archive_or_default(ppoa_archive or ppob_archive)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    apply_plot_theme(fig, ax)
    if FILL_FIGURE_CANVAS:
        ax.set_position([0.0, 0.0, 1.0, 0.95])

    all_times_for_moon = []

    # Main trajectories
    if ppoa_archive is not None:
        traj_a, t_a = get_traj_and_time(ppoa_archive)
        all_times_for_moon.append(t_a)
        xyz_a = convert_path_to_plot_frame(traj_a, t_a, frame, mu)
        plot_path(ax, xyz_a, PPOA_COLOR, PPOA_LINEWIDTH, "PPO-A trajectory",
          t_hist=t_a, frame=frame, mu=mu)

    gap_traj = np.zeros((0, 4))
    gap_t = np.zeros((0,))

    if (
        MODE == "compare"
        and ppoa_archive is not None
        and ppob_archive is not None
        and FILL_PPOA_TO_PPOB_GAP_IN_COMPARE
    ):
        gap_traj, gap_t = propagate_ballistic_gap_from_ppoa(
            ppoa_archive=ppoa_archive,
            gap_minutes=PPOA_TO_PPOB_GAP_MINUTES,
            extra_minutes=PPOA_TO_PPOB_EXTRA_GAP_MINUTES,
            mu=mu,
        )
        if len(gap_traj) > 0:
            all_times_for_moon.append(gap_t)
            xyz_gap = convert_path_to_plot_frame(gap_traj, gap_t, frame, mu)
            plot_path(ax, xyz_gap, GAP_COLOR, GAP_LINEWIDTH, "PPO-A ballistic gap", GAP_LINESTYLE,
                t_hist=gap_t, frame=frame, mu=mu)

            traj_b_raw, t_b_raw = get_traj_and_time(ppob_archive)
            if len(traj_b_raw) > 0:
                mismatch = traj_b_raw[0, :2] - gap_traj[-1, :2]
                mismatch_norm = float(np.linalg.norm(mismatch))
                if PRINT_GAP_DIAGNOSTICS:
                    print("\n" + "=" * 90)
                    print("PPO-A -> PPO-B GAP DIAGNOSTICS")
                    print("=" * 90)
                    print(f"Gap propagation time      : {PPOA_TO_PPOB_GAP_MINUTES + PPOA_TO_PPOB_EXTRA_GAP_MINUTES:.3f} min")
                    print(f"Gap end rotating xy       : {gap_traj[-1, :2]}")
                    print(f"PPO-B start rotating xy   : {traj_b_raw[0, :2]}")
                    print(f"Rotating-frame mismatch   : {mismatch_norm:.6e} nondim")
                    print("If this is large, the selected PPO-A and PPO-B archives are probably not the same physical case.")
                    print("=" * 90 + "\n")

                if VISUALLY_SNAP_PPOB_START_TO_GAP_END:
                    ppob_archive = maybe_shift_archive_for_visual_snap(ppob_archive, -mismatch)

    if ppob_archive is not None:
        traj_b, t_b = get_traj_and_time(ppob_archive)
        all_times_for_moon.append(t_b)
        xyz_b = convert_path_to_plot_frame(traj_b, t_b, frame, mu)
        plot_path(ax, xyz_b, PPOB_COLOR, PPOB_LINEWIDTH, "PPO-B trajectory",
          t_hist=t_b, frame=frame, mu=mu)

    # Ballistic references
    if SHOW_BALLISTIC:
        if ppoa_archive is not None:
            bal_a, tb_a = get_ballistic_ref_and_time(ppoa_archive)
            if len(bal_a) > 0:
                all_times_for_moon.append(tb_a)
                xyz = convert_path_to_plot_frame(bal_a, tb_a, frame, mu)
                plot_path(ax, xyz, BALLISTIC_PPOA_COLOR, BALLISTIC_LINEWIDTH, "PPO-A ballistic", BALLISTIC_LINESTYLE,
                    t_hist=tb_a, frame=frame, mu=mu)

        if ppob_archive is not None:
            bal_b, tb_b = get_ballistic_ref_and_time(ppob_archive)
            if len(bal_b) > 0:
                all_times_for_moon.append(tb_b)
                xyz = convert_path_to_plot_frame(bal_b, tb_b, frame, mu)
                plot_path(ax, xyz, BALLISTIC_PPOB_COLOR, BALLISTIC_LINEWIDTH, "PPO-B ballistic", BALLISTIC_LINESTYLE,
                    t_hist=tb_b, frame=frame, mu=mu)

    # DV arrows
    if SHOW_DV_ARROWS:
        if ppoa_archive is not None:
            plot_dv_arrows(ax, ppoa_archive, DV_ARROW_PPOA_COLOR, DV_ARROW_LENGTH_SCALE_PPOA, frame, mu)
        if ppob_archive is not None:
            plot_dv_arrows(ax, ppob_archive, DV_ARROW_PPOB_COLOR, DV_ARROW_LENGTH_SCALE_PPOB, frame, mu)

    # Moon trail in inertial frame
    if all_times_for_moon:
        all_t = np.concatenate([np.asarray(t, dtype=float).reshape(-1) for t in all_times_for_moon if len(t) > 0])
    else:
        all_t = np.zeros((0,), dtype=float)

    plot_moon_trail(ax, all_t, frame, mu)

    # Draw Earth and current/reference Moon sphere
    earth_xyz = earth_position_in_plot_frame(frame, mu)
    draw_sphere(ax, earth_xyz, EARTH_RADIUS_ND, EARTH_COLOR, EARTH_ALPHA)

    # For rotating, Moon fixed. For inertial, draw Moon at final plotted time.
    if frame == "rotating":
        moon_xy = moon_position_in_plot_frame(np.asarray([0.0]), frame, mu)[0]
    else:
        t_moon_now = float(np.nanmax(all_t)) if len(all_t) > 0 else 0.0
        moon_xy = moon_position_in_plot_frame(np.asarray([t_moon_now]), frame, mu)[0]

    draw_sphere(ax, np.array([moon_xy[0], moon_xy[1], 0.0]), MOON_RADIUS_ND, MOON_COLOR, MOON_ALPHA)

    # Ghost Moon at closest approach
    if SHOW_GHOST_MOON_AT_CLOSEST_APPROACH:
        candidates = []
        if ppoa_archive is not None:
            traj_a, t_a = get_traj_and_time(ppoa_archive)
            candidates.append((traj_a, t_a, "PPO-A"))
        if ppob_archive is not None:
            traj_b, t_b = get_traj_and_time(ppob_archive)
            candidates.append((traj_b, t_b, "PPO-B"))
        if len(gap_traj) > 0:
            candidates.append((gap_traj, gap_t, "gap"))

        best = None
        for traj, th, name in candidates:
            moon_pos, tc, dc = find_closest_moon_event(traj, th, frame, mu)
            if best is None or dc < best[0]:
                best = (dc, moon_pos, tc, name)

        if best is not None and np.isfinite(best[0]):
            _, moon_pos, tc, name = best
            draw_sphere(ax, moon_pos, MOON_RADIUS_ND, GHOST_MOON_COLOR, GHOST_MOON_ALPHA)

            if SHOW_GHOST_MOON_TEXT:
                ax.text(
                    moon_pos[0], moon_pos[1], moon_pos[2] + 2.5 * MOON_RADIUS_ND,
                    f"Ghost Moon\nclosest {name}",
                    fontsize=max(8, TICK_SIZE - 1),
                    color="gray",
                )

    apply_visual_settings(ax)

    if SAVE_FIGURE:
        output_path = make_output_path()
        transparent = PLOT_THEME == "transparent"

        plt.savefig(
            output_path,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=transparent,
            facecolor="none" if transparent else fig.get_facecolor(),
        )

        print(f"Saved figure: {output_path.resolve()}")

    
    if FULLSCREEN_WINDOW:
        manager = plt.get_current_fig_manager()
        try:
            manager.window.showMaximized()
        except Exception:
            try:
                manager.full_screen_toggle()
            except Exception:
                pass

    plt.show()


def main():
    mode = MODE.lower().strip()
    if mode not in ("ppoa", "ppob", "compare"):
        raise ValueError("MODE must be 'ppoa', 'ppob', or 'compare'.")

    ppoa_archive = None
    ppob_archive = None

    if mode == "ppoa":
        report = select_episode_report("PPO-A") if MANUAL_SELECT else _as_path(PPOA_EPISODE_REPORT_JSON)
        ppoa_archive = load_archive(report)

    elif mode == "ppob":
        report = select_episode_report("PPO-B") if MANUAL_SELECT else _as_path(PPOB_EPISODE_REPORT_JSON)
        ppob_archive = load_archive(report)

    elif mode == "compare":
        if MANUAL_SELECT:
            report_a = select_episode_report("PPO-A")
            report_b = select_episode_report("PPO-B")
        else:
            report_a = _as_path(PPOA_EPISODE_REPORT_JSON)
            report_b = _as_path(PPOB_EPISODE_REPORT_JSON)

        ppoa_archive = load_archive(report_a)
        ppob_archive = load_archive(report_b)

    make_money_plot(ppoa_archive, ppob_archive)


if __name__ == "__main__":
    main()
