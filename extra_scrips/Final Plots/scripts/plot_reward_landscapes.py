"""
Thesis-ready reward landscape generator.

Reads project modules from:
    Final plotting/project_modules/

Outputs:
    Final plotting/outputs/thesis_ready/reward_landscapes/
"""

from pathlib import Path
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm
from style.thesis_style import ieee_title


# ============================================================
# PATH SETUP
# ============================================================

FINAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FINAL_ROOT.parent
MODULE_DIR = FINAL_ROOT / "project_modules"

sys.path.insert(0, str(FINAL_ROOT))
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


from style.thesis_style import (
    apply_thesis_style,
    get_figsize,
    save_thesis_figure,
    clean_axis,
)

from config import CR3BPConfig, RewardConfig
from cr3bp_env_v4 import (
    SeanStyleReward,
    apply_stage_to_cfg,
    earth_moon_positions,
    kms_to_nondim_dv,
)
from curriculum_ppoa import build_curriculum_ppoa
from curriculum_ppob import build_curriculum_ppob


# ============================================================
# USER SETTINGS
# ============================================================

GENERATE_PPO_TLI = True
GENERATE_PPO_MCC = True

# None = all stages.
# Example: [0, 1, 2]
STAGE_FILTER_TLI = None
STAGE_FILTER_MCC = None

GENERATE_PRE_NO_INVALID = True
GENERATE_PRE_WITH_INVALID = True
GENERATE_POST = True

GRID_NX = 1200
GRID_NY = 800

X_MIN = -0.25
X_MAX = 1.15
Y_MIN = -0.50
Y_MAX = 0.50

CMAP = "RdBu_r"

SHOW_TITLE = True
SHOW_GRID = False
SHOW_LEGEND = False

SHOW_CONTOURS = True
CONTOUR_LEVELS = [-200, -100, -50, 0, 50, 100, 150, 200]
CONTOUR_LINEWIDTH = 0.45
CONTOUR_ALPHA = 0.35
SHOW_ZERO_CONTOUR = True
ZERO_CONTOUR_LINEWIDTH = 1.0

SHOW_EARTH = False
SHOW_MOON = False
SHOW_LEO_ORBIT = False
SHOW_MOON_FLYBY_BOUND = True
SHOW_RETURN_CORRIDOR = True

COLORBAR_LABEL = "Reward contribution"

REPLACE_NONFINITE_VALUES = True
NONFINITE_REPLACEMENT = 0.0

PENALTY_MODE = "overwrite"
INVALID_OVERWRITE = True

INCLUDE_EARTH_IMPACT = True
INCLUDE_MOON_IMPACT = True
INCLUDE_ESCAPE = True

POST_INCLUDE_FLYBY_REWARD = False
PRE_INCLUDE_RETURN_REWARD = False


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_stem(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def sanitize_field(Z):
    Z = np.asarray(Z, dtype=float)
    if REPLACE_NONFINITE_VALUES:
        Z = np.nan_to_num(
            Z,
            nan=NONFINITE_REPLACEMENT,
            posinf=NONFINITE_REPLACEMENT,
            neginf=NONFINITE_REPLACEMENT,
        )
    return Z


def apply_overrides_to_cfg(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).get("env", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def apply_overrides_to_reward_cfg(reward_cfg, overrides):
    reward_cfg = copy.deepcopy(reward_cfg)
    for key, value in (overrides or {}).get("reward", {}).items():
        if hasattr(reward_cfg, key):
            setattr(reward_cfg, key, value)
    return reward_cfg


def load_profile(profile_key):
    base_cfg = CR3BPConfig()
    base_reward_cfg = RewardConfig()

    if profile_key == "PPO_TLI":
        curriculum, overrides = build_curriculum_ppoa(kms_to_nondim_dv)
    elif profile_key == "PPO_MCC":
        curriculum, overrides = build_curriculum_ppob()
    else:
        raise ValueError(profile_key)

    base_cfg = apply_overrides_to_cfg(base_cfg, overrides)
    reward_cfg = apply_overrides_to_reward_cfg(base_reward_cfg, overrides)

    return curriculum, base_cfg, reward_cfg


def build_stage_objects(profile_key, stage):
    _, base_cfg, reward_cfg = load_profile(profile_key)
    cfg = apply_stage_to_cfg(base_cfg, stage)
    reward_model = SeanStyleReward(reward_cfg, stage.reward_weights)
    return cfg, reward_model


def make_grid():
    x = np.linspace(X_MIN, X_MAX, GRID_NX)
    y = np.linspace(Y_MIN, Y_MAX, GRID_NY)
    return np.meshgrid(x, y)


def distance_fields(cfg, X, Y):
    rE_pos, rM_pos = earth_moon_positions(cfg.mu)

    rE = np.sqrt((X - rE_pos[0]) ** 2 + (Y - rE_pos[1]) ** 2)
    rM = np.sqrt((X - rM_pos[0]) ** 2 + (Y - rM_pos[1]) ** 2)
    rB = np.sqrt(X ** 2 + Y ** 2)

    return rE, rM, rB, rE_pos, rM_pos


# ============================================================
# REWARD MAP MODELS
# ============================================================

def flyby_reward_array(reward_model, cfg, rM):
    rmin = np.asarray(rM, dtype=float)

    d0 = float(reward_model.cfg.r0_distance_flyby)
    beta = float(reward_model.cfg.beta_distance_flyby)
    w = float(reward_model.w.w_flyby)

    rf = float(getattr(cfg, "r_moon_flyby", reward_model.cfg.moon_radius))

    d_eff = np.maximum(rmin, rf)
    x = np.clip(d_eff / max(d0, 1e-12), 0.0, 100.0)
    rd = 1.0 / (1.0 + x ** beta)

    x_rf = np.clip(rf / max(d0, 1e-12), 0.0, 100.0)
    rd_rf = 1.0 / (1.0 + x_rf ** beta)

    return sanitize_field(w * rd / max(rd_rf, 1e-12))


def return_reward_array(reward_model, cfg, rE):
    r = np.asarray(rE, dtype=float)

    rp_min = float(cfg.rp_min)
    rp_max = float(cfg.rp_max)

    d = np.zeros_like(r)
    d = np.where(r < rp_min, rp_min - r, d)
    d = np.where(r > rp_max, r - rp_max, d)

    beta = float(reward_model.cfg.beta_distance_return)
    d0 = float(reward_model.cfg.r0_distance_return)
    w = float(reward_model.w.w_return)

    x = np.clip(d / max(d0, 1e-12), 0.0, 100.0)
    rr = 1.0 / (1.0 + x ** beta)

    return sanitize_field(w * rr)


def invalid_preflyby_return_mask(cfg, rE, rM):
    if not bool(getattr(cfg, "ballistic_invalid_preflyby_return_enabled", True)):
        return np.zeros_like(rE, dtype=bool)

    arm_rE = float(getattr(cfg, "ballistic_invalid_return_arm_rE", np.inf))
    moon_far_rM = float(getattr(cfg, "ballistic_invalid_return_moon_far_rM", np.inf))

    return (rE >= arm_rE) & (rM > moon_far_rM)


def apply_mask_penalty(Z, mask, penalty_value, overwrite):
    out = np.array(Z, copy=True)
    if overwrite:
        out[mask] = penalty_value
    else:
        out[mask] += penalty_value
    return sanitize_field(out)


def apply_crash_escape_penalties(Z, cfg, reward_model, rE, rM, rB, phase):
    out = np.array(Z, copy=True)
    overwrite = PENALTY_MODE.lower() == "overwrite"

    if INCLUDE_ESCAPE:
        out = apply_mask_penalty(
            out,
            rB >= float(cfg.r_escape),
            -float(reward_model.w.w_escape),
            overwrite,
        )

    if INCLUDE_EARTH_IMPACT:
        if phase == "post":
            penalty = -float(reward_model.w.w_postflyby_earth_crash)
        else:
            penalty = -float(reward_model.w.w_earth_crash)

        out = apply_mask_penalty(
            out,
            rE <= float(cfg.r_earth_impact),
            penalty,
            overwrite,
        )

    if INCLUDE_MOON_IMPACT:
        out = apply_mask_penalty(
            out,
            rM <= float(cfg.r_moon_impact),
            -float(reward_model.w.w_moon_crash),
            overwrite,
        )

    return sanitize_field(out)


def build_reward_map(cfg, reward_model, phase, invalid_enabled):
    X, Y = make_grid()
    rE, rM, rB, rE_pos, rM_pos = distance_fields(cfg, X, Y)

    Z = np.zeros_like(X, dtype=float)

    if phase == "pre":
        Z += flyby_reward_array(reward_model, cfg, rM)

        if PRE_INCLUDE_RETURN_REWARD:
            Z += return_reward_array(reward_model, cfg, rE)

        if invalid_enabled:
            invalid = invalid_preflyby_return_mask(cfg, rE, rM)
            Z = apply_mask_penalty(
                Z,
                invalid,
                -float(reward_model.w.w_invalid_preflyby_earth_return),
                overwrite=INVALID_OVERWRITE,
            )

    elif phase == "post":
        Z += return_reward_array(reward_model, cfg, rE)

        if POST_INCLUDE_FLYBY_REWARD:
            Z += flyby_reward_array(reward_model, cfg, rM)

    else:
        raise ValueError("phase must be 'pre' or 'post'.")

    Z = apply_crash_escape_penalties(Z, cfg, reward_model, rE, rM, rB, phase)
    return X, Y, sanitize_field(Z), rE_pos, rM_pos


# ============================================================
# PLOTTING HELPERS
# ============================================================

def robust_limits(Z):
    vals = np.asarray(Z, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return -1.0, 1.0

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    if abs(vmax - vmin) < 1e-12:
        pad = max(abs(vmax), 1.0) * 0.05
        return vmin - pad, vmax + pad

    return vmin, vmax


def add_circle(ax, xy, radius, color, fill=False, linestyle="-", linewidth=1.0, alpha=1.0):
    ax.add_patch(Circle(
        xy,
        radius,
        facecolor=color if fill else "none",
        edgecolor=color,
        fill=fill,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=5,
    ))


def add_overlays(ax, cfg, rE_pos, rM_pos):
    if SHOW_EARTH:
        add_circle(ax, rE_pos, cfg.r_earth_impact, "tab:blue", fill=True)

    if SHOW_MOON:
        add_circle(ax, rM_pos, cfg.r_moon_impact, "gray", fill=True)

    if SHOW_LEO_ORBIT:
        add_circle(ax, rE_pos, cfg.r0_earth, "lightskyblue", linestyle=":")

    if SHOW_MOON_FLYBY_BOUND:
        add_circle(ax, rM_pos, cfg.r_moon_flyby, "black", linestyle=":", linewidth=1.0)

    if SHOW_RETURN_CORRIDOR:
        add_circle(ax, rE_pos, cfg.rp_min, "black", linestyle="--", linewidth=0.9)
        add_circle(ax, rE_pos, cfg.rp_max, "black", linestyle="-.", linewidth=0.9)


def make_title(profile_key, stage_idx, phase, invalid_enabled):
    profile = "PPO-TLI" if profile_key == "PPO_TLI" else "PPO-MCC"

    if phase == "pre" and invalid_enabled:
        desc = "Pre-flyby reward landscape with invalid-return region"
    elif phase == "pre":
        desc = "Pre-flyby reward landscape"
    else:
        desc = "Post-flyby return-corridor reward landscape"

    return f"{desc}\n{profile}, stage {stage_idx + 1}"


def make_colorbar_ticks(vmin, vmax):
    """
    Five perceptually/evenly placed colorbar ticks for TwoSlopeNorm:

        vmin
        midpoint between vmin and 0 in colorbar space
        0
        midpoint between 0 and vmax in colorbar space
        vmax

    The intermediate values are computed in normalized colorbar space,
    then converted back to data values.
    """

    vmin = float(vmin)
    vmax = float(vmax)

    if not (vmin < 0.0 < vmax):
        return [int(round(vmin)), int(round((vmin + vmax) / 2)), int(round(vmax))]

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    normalized_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    ticks = []
    for p in normalized_positions:
        value = float(norm.inverse(p))
        ticks.append(int(round(value)))

    return ticks


def plot_heatmap(X, Y, Z, cfg, rE_pos, rM_pos, profile_key, stage_idx, phase, invalid_enabled, out_path):
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=get_figsize("double"))

    vmin, vmax = robust_limits(Z)

    extent = [
        float(np.min(X)),
        float(np.max(X)),
        float(np.min(Y)),
        float(np.max(Y)),
    ]

    im = ax.imshow(
        Z,
        extent=extent,
        origin="lower",
        cmap=CMAP,
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
        aspect="equal",
    )

    Z_contour = np.array(Z, copy=True)

    rE = np.sqrt((X - rE_pos[0]) ** 2 + (Y - rE_pos[1]) ** 2)
    rM = np.sqrt((X - rM_pos[0]) ** 2 + (Y - rM_pos[1]) ** 2)

    Z_contour[rE <= cfg.r_earth_impact] = np.nan
    Z_contour[rM <= cfg.r_moon_impact] = np.nan

    if SHOW_CONTOURS:
        levels = [lvl for lvl in CONTOUR_LEVELS if vmin < lvl < vmax]
        if levels:
            cs = ax.contour(
                X,
                Y,
                Z_contour,
                levels=levels,
                colors="black",
                linewidths=CONTOUR_LINEWIDTH,
                alpha=CONTOUR_ALPHA,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

    if SHOW_ZERO_CONTOUR and vmin < 0.0 < vmax:
        ax.contour(
            X,
            Y,
            Z_contour,
            levels=[0.0],
            colors="black",
            linewidths=ZERO_CONTOUR_LINEWIDTH,
            alpha=0.85,
        )

    add_overlays(ax, cfg, rE_pos, rM_pos)

    if SHOW_TITLE:
        ax.set_title(ieee_title(make_title(profile_key, stage_idx, phase, invalid_enabled)))

    ax.set_xlabel(r"$x$ rotating frame [nondim]")
    ax.set_ylabel(r"$y$ rotating frame [nondim]")

    clean_axis(ax, grid=SHOW_GRID)

    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label(COLORBAR_LABEL)

    ticks = make_colorbar_ticks(vmin, vmax)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0f}" for t in ticks])

    if SHOW_LEGEND:
        ax.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    save_thesis_figure(fig, out_path)
    plt.close(fig)


# ============================================================
# GENERATION
# ============================================================

def save_stage_summary(stage_dir, profile_key, stage_idx, stage, cfg, reward_model):
    lines = [
        "REWARD LANDSCAPE SUMMARY",
        "=" * 72,
        f"profile: {profile_key}",
        f"stage_index: {stage_idx}",
        f"stage_name: {stage.name}",
        "",
        "GEOMETRY",
        f"mu: {cfg.mu}",
        f"r_earth_impact: {cfg.r_earth_impact}",
        f"r_moon_impact: {cfg.r_moon_impact}",
        f"r_moon_flyby: {cfg.r_moon_flyby}",
        f"rp_min: {cfg.rp_min}",
        f"rp_max: {cfg.rp_max}",
        f"r_escape: {cfg.r_escape}",
        "",
        "REWARD WEIGHTS",
        f"w_flyby: {reward_model.w.w_flyby}",
        f"w_return: {reward_model.w.w_return}",
        f"w_escape: {reward_model.w.w_escape}",
        f"w_earth_crash: {reward_model.w.w_earth_crash}",
        f"w_moon_crash: {reward_model.w.w_moon_crash}",
        f"w_postflyby_earth_crash: {reward_model.w.w_postflyby_earth_crash}",
        f"w_invalid_preflyby_earth_return: {reward_model.w.w_invalid_preflyby_earth_return}",
    ]

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "reward_landscape_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def generate_for_stage(profile_key, stage_idx, stage, output_dir):
    cfg, reward_model = build_stage_objects(profile_key, stage)

    stage_dir = (
        Path(output_dir)
        / "reward_landscapes"
        / profile_key
        / f"stage_{stage_idx:02d}_{safe_stem(stage.name)}"
    )
    stage_dir.mkdir(parents=True, exist_ok=True)

    save_stage_summary(stage_dir, profile_key, stage_idx, stage, cfg, reward_model)

    jobs = []

    if GENERATE_PRE_NO_INVALID:
        jobs.append(("pre", False, "pre_flyby_no_invalid"))

    if GENERATE_PRE_WITH_INVALID:
        jobs.append(("pre", True, "pre_flyby_with_invalid"))

    if GENERATE_POST:
        jobs.append(("post", False, "post_flyby"))

    for phase, invalid_enabled, name in jobs:
        X, Y, Z, rE_pos, rM_pos = build_reward_map(cfg, reward_model, phase, invalid_enabled)

        out_path = stage_dir / name

        plot_heatmap(
            X,
            Y,
            Z,
            cfg,
            rE_pos,
            rM_pos,
            profile_key,
            stage_idx,
            phase,
            invalid_enabled,
            out_path,
        )

        print(f"[OK] {profile_key} stage {stage_idx:02d}: {name}")


def generate_profile(profile_key, output_dir, stage_filter):
    curriculum, _, _ = load_profile(profile_key)

    if stage_filter is None:
        indices = list(range(len(curriculum)))
    else:
        indices = [i for i in stage_filter if 0 <= i < len(curriculum)]

    for i in indices:
        generate_for_stage(profile_key, i, curriculum[i], output_dir)


# ============================================================
# ENTRY POINT FOR plot_all.py
# ============================================================

def main(data_dir=None, output_dir=None):
    if output_dir is None:
        output_dir = FINAL_ROOT / "outputs" / "thesis_ready"

    output_dir = Path(output_dir)

    print("=" * 72)
    print("THESIS REWARD LANDSCAPE GENERATION")
    print("=" * 72)
    print(f"Module folder : {MODULE_DIR}")
    print(f"Output folder : {output_dir}")
    print(f"Grid          : {GRID_NX} x {GRID_NY}")
    print("=" * 72)

    if GENERATE_PPO_TLI:
        print("\nGenerating PPO-TLI reward landscapes...")
        generate_profile("PPO_TLI", output_dir, STAGE_FILTER_TLI)

    if GENERATE_PPO_MCC:
        print("\nGenerating PPO-MCC reward landscapes...")
        generate_profile("PPO_MCC", output_dir, STAGE_FILTER_MCC)

    print("\nDone.")


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":
    main()