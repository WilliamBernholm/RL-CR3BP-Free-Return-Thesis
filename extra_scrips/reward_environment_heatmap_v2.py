"""
reward_environment_heatmap_batch.py

Batch reward-landscape visualizer for the CR3BP RL project.

Generates:
- PPO_TLI / all curriculum stages
- PPO_MCC / all curriculum stages
- pre_flyby_no_invalid
- pre_flyby_with_invalid
- post_flyby

Output:
reward_heatmaps_batch/
    PPO_TLI/stage_00_<stage_name>/*.png, *.pdf
    PPO_MCC/stage_00_<stage_name>/*.png, *.pdf
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm

from config import CR3BPConfig, RewardConfig
from cr3bp_env_v4 import SeanStyleReward, apply_stage_to_cfg, earth_moon_positions, kms_to_nondim_dv
from curriculum_ppoa import build_curriculum_ppoa
from curriculum_ppob import build_curriculum_ppob


# ============================================================
# USER SETTINGS
# ============================================================

OUT_DIR = Path("reward_heatmaps_batch")
SAVE_PNG = True
SAVE_PDF = False
SHOW_FIGURES = False

GENERATE_PPO_TLI = True
GENERATE_PPO_MCC = True
STAGE_FILTER = None  # None = all stages, or e.g. [0, 2]

GENERATE_PRE_NO_INVALID = True
GENERATE_PRE_WITH_INVALID = True
GENERATE_POST = True

X_MIN = -0.25
X_MAX = 1.15
Y_MIN = -0.5
Y_MAX = 0.5
GRID_NX = 3000
GRID_NY = 2000

CMAP = "RdBu_r"
AUTO_COLOR_LIMITS = True
ROBUST_PERCENTILE = 95.0
VMIN = -220.0
VMAX = 220.0

# Fixes missing/blank strips caused by NaN/Inf values.
REPLACE_NONFINITE_VALUES = True
NONFINITE_REPLACEMENT = 0.0

FIGSIZE = (16, 6)
DPI = 500
TITLE_SIZE = 21
LABEL_SIZE = 19
TICK_SIZE = 16
COLORBAR_LABEL_SIZE = 18
COLORBAR_TICK_SIZE = 15
LEGEND_SIZE = 12

X_LABEL = r"$x$ rotating frame [nondim]"
Y_LABEL = r"$y$ rotating frame [nondim]"
COLORBAR_LABEL = "Reward contribution"

SHOW_GRID = False
SHOW_LEGEND = True

SHOW_CONTOURS = True
CONTOUR_LEVELS = [-200, -150, -100, -50, 0, 25, 50, 75, 100, 150, 200]
CONTOUR_LINEWIDTH = 0.60
CONTOUR_ALPHA = 0.45
SHOW_ZERO_CONTOUR = True
ZERO_CONTOUR_LINEWIDTH = 1.35

SHOW_EARTH = False
SHOW_MOON = False
SHOW_LEO_ORBIT = False
SHOW_MOON_FLYBY_BOUND = True
SHOW_RETURN_CORRIDOR = True
SHOW_INVALID_BOUNDARIES = False
SHOW_ESCAPE_BOUNDARY = False

FILTER_CONTOUR_LABELS_NEAR_EARTH = True
CONTOUR_LABEL_EARTH_CLEARANCE = 0.09

EARTH_COLOR = "dodgerblue"
MOON_COLOR = "gray"
LEO_COLOR = "lightskyblue"
FLYBY_COLOR = "black"
RETURN_COLOR = "black"
INVALID_COLOR = "purple"
ESCAPE_COLOR = "purple"

EARTH_ALPHA = 0.88
MOON_ALPHA = 0.88
OVERLAY_LINEWIDTH = 1.35

# "overwrite" is clearer for terminal regions.
PENALTY_MODE = "overwrite"  # "overwrite" or "add"
INVALID_OVERWRITE = True

INCLUDE_EARTH_IMPACT = True
INCLUDE_MOON_IMPACT = True
INCLUDE_ESCAPE = True

POST_INCLUDE_FLYBY_REWARD = False
PRE_INCLUDE_RETURN_REWARD = False

SAVE_STAGE_SUMMARY_TXT = True


# ============================================================
# HELPERS
# ============================================================

def safe_stem(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(s))


def sanitize_field(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    if REPLACE_NONFINITE_VALUES:
        Z = np.nan_to_num(Z, nan=NONFINITE_REPLACEMENT,
                          posinf=NONFINITE_REPLACEMENT,
                          neginf=NONFINITE_REPLACEMENT)
    return Z


def apply_overrides_to_cfg(cfg: CR3BPConfig, overrides: dict) -> CR3BPConfig:
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).get("env", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def apply_overrides_to_reward_cfg(reward_cfg: RewardConfig, overrides: dict) -> RewardConfig:
    reward_cfg = copy.deepcopy(reward_cfg)
    for key, value in (overrides or {}).get("reward", {}).items():
        if hasattr(reward_cfg, key):
            setattr(reward_cfg, key, value)
    return reward_cfg


def load_profile(profile_key: str):
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
    return curriculum, base_cfg, reward_cfg, overrides


def build_stage_objects(profile_key: str, stage):
    _, base_cfg, reward_cfg, _ = load_profile(profile_key)
    cfg = apply_stage_to_cfg(base_cfg, stage)
    reward_model = SeanStyleReward(reward_cfg, stage.reward_weights)
    return cfg, reward_cfg, reward_model


def make_grid() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(X_MIN, X_MAX, GRID_NX)
    y = np.linspace(Y_MIN, Y_MAX, GRID_NY)
    return np.meshgrid(x, y)


def distance_fields(cfg: CR3BPConfig, X: np.ndarray, Y: np.ndarray):
    rE_pos, rM_pos = earth_moon_positions(cfg.mu)
    rE_pos = np.asarray(rE_pos, dtype=float)
    rM_pos = np.asarray(rM_pos, dtype=float)
    rE = np.sqrt((X - rE_pos[0])**2 + (Y - rE_pos[1])**2)
    rM = np.sqrt((X - rM_pos[0])**2 + (Y - rM_pos[1])**2)
    rB = np.sqrt(X**2 + Y**2)
    return rE, rM, rB, rE_pos, rM_pos


# ============================================================
# REWARD FIELD MODELS
# ============================================================

def flyby_reward_array(reward_model: SeanStyleReward, cfg: CR3BPConfig, rM: np.ndarray) -> np.ndarray:
    rmin = np.asarray(rM, dtype=float)
    d0 = float(reward_model.cfg.r0_distance_flyby)
    beta = float(reward_model.cfg.beta_distance_flyby)
    w = float(reward_model.w.w_flyby)
    rf = float(getattr(cfg, "r_moon_flyby", reward_model.cfg.moon_radius))

    # Matches reward class: inside flyby bound saturates at rf.
    d_eff = np.maximum(rmin, rf)
    x = np.clip(d_eff / max(d0, 1e-12), 0.0, 100.0)
    rd = 1.0 / (1.0 + x**beta)

    x_rf = np.clip(rf / max(d0, 1e-12), 0.0, 100.0)
    rd_rf = 1.0 / (1.0 + x_rf**beta)

    return sanitize_field(w * (rd / max(rd_rf, 1e-12)))


def return_reward_array(reward_model: SeanStyleReward, cfg: CR3BPConfig, rE: np.ndarray) -> np.ndarray:
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
    rr = 1.0 / (1.0 + x**beta)
    return sanitize_field(w * rr)


def invalid_preflyby_return_mask(cfg: CR3BPConfig, rE: np.ndarray, rM: np.ndarray) -> np.ndarray:
    """
    2D diagnostic projection of invalid pre-flyby Earth-return logic.

    This is intentionally NOT the exact history-dependent environment event.
    It is a report/diagnostic slice.

    Assumptions:
    - each (x,y) is treated as a hypothetical closest lunar approach point
    - pre-flyby state is assumed
    - Earth-relative radial velocity is assumed to be at a turning/fallback condition:
          vrE <= 0

    A point is marked invalid if:
    - it is beyond the Earth-distance arming radius
    - it is still far from the Moon
    - it is not outbound in the Earth-radial sense
    """
    if not bool(getattr(cfg, "ballistic_invalid_preflyby_return_enabled", True)):
        return np.zeros_like(rE, dtype=bool)

    arm_rE = float(getattr(cfg, "ballistic_invalid_return_arm_rE", np.inf))
    moon_far_rM = float(getattr(cfg, "ballistic_invalid_return_moon_far_rM", np.inf))

    # Diagnostic assumption, not exact env threshold:
    # vrE = 0 is treated as turning/fallback.
    vrE_assumed = 0.0

    invalid = (
        (rE >= arm_rE)
        & (rM > moon_far_rM)
        & (vrE_assumed <= 0.0)
    )

    return invalid


def apply_mask_penalty(Z: np.ndarray, mask: np.ndarray, penalty_value: float, overwrite: bool) -> np.ndarray:
    out = np.array(Z, copy=True)
    if overwrite:
        out[mask] = penalty_value
    else:
        out[mask] += penalty_value
    return sanitize_field(out)


def apply_crash_escape_penalties(Z, cfg, reward_model, rE, rM, rB, phase):
    out = np.array(Z, copy=True)
    overwrite = PENALTY_MODE.lower().strip() == "overwrite"

    if INCLUDE_ESCAPE:
        out = apply_mask_penalty(out, rB >= float(cfg.r_escape), -float(reward_model.w.w_escape), overwrite)
    if INCLUDE_EARTH_IMPACT:
        if phase == "post":
            earth_crash_penalty = -float(reward_model.w.w_postflyby_earth_crash)
        else:
            earth_crash_penalty = -float(reward_model.w.w_earth_crash)

        out = apply_mask_penalty(
            out,
            rE <= float(cfg.r_earth_impact),
            earth_crash_penalty,
            overwrite,
        )
    if INCLUDE_MOON_IMPACT:
        out = apply_mask_penalty(out, rM <= float(cfg.r_moon_impact), -float(reward_model.w.w_moon_crash), overwrite)

    return sanitize_field(out)


def build_reward_map(cfg, reward_model, phase: str, invalid_enabled: bool):
    X, Y = make_grid()
    rE, rM, rB, rE_pos, rM_pos = distance_fields(cfg, X, Y)
    Z = np.zeros_like(X, dtype=float)

    if phase == "pre":
        # Interpret each point as hypothetical closest lunar approach.
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
        # Interpret each point as hypothetical closest post-flyby Earth approach.
        Z += return_reward_array(reward_model, cfg, rE)

        if POST_INCLUDE_FLYBY_REWARD:
            Z += flyby_reward_array(reward_model, cfg, rM)

    else:
        raise ValueError("phase must be 'pre' or 'post'.")

    Z = apply_crash_escape_penalties(Z, cfg, reward_model, rE, rM, rB, phase)
    return X, Y, sanitize_field(Z), rE_pos, rM_pos


# ============================================================
# PLOTTING
# ============================================================

def robust_limits(Z):
    vals = np.asarray(Z, dtype=float)
    vals = vals[np.isfinite(vals)].ravel()

    if vals.size == 0:
        return VMIN, VMAX

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    # Safety if all values are identical
    if abs(vmax - vmin) < 1e-12:
        pad = max(abs(vmax), 1.0) * 0.05
        return vmin - pad, vmax + pad

    return vmin, vmax


def add_circle(ax, xy, radius, color, label=None, fill=False, alpha=1.0, linestyle="-", linewidth=1.5):
    ax.add_patch(Circle(
        xy, radius,
        facecolor=color if fill else "none",
        edgecolor="none",
        alpha=alpha,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        zorder=5,
    ))


def add_overlays(ax, cfg, rE_pos, rM_pos, phase, invalid_enabled):
    if SHOW_EARTH:
        add_circle(ax, rE_pos, cfg.r_earth_impact, EARTH_COLOR,
                   label=f"Earth impact r={cfg.r_earth_impact:.4f}",
                   fill=True, alpha=EARTH_ALPHA, linewidth=OVERLAY_LINEWIDTH)

    if SHOW_MOON:
        add_circle(ax, rM_pos, cfg.r_moon_impact, MOON_COLOR,
                   label=f"Moon impact r={cfg.r_moon_impact:.4f}",
                   fill=True, alpha=MOON_ALPHA, linewidth=OVERLAY_LINEWIDTH)

    if SHOW_LEO_ORBIT:
        add_circle(ax, rE_pos, cfg.r0_earth, LEO_COLOR,
                   label=f"400 km LEO r={cfg.r0_earth:.4f}",
                   fill=False, linestyle=":", linewidth=OVERLAY_LINEWIDTH)

    if SHOW_MOON_FLYBY_BOUND:
        add_circle(ax, rM_pos, cfg.r_moon_flyby, FLYBY_COLOR,
                   label=f"Flyby bound r={cfg.r_moon_flyby:.4f}",
                   fill=False, linestyle=":", linewidth=OVERLAY_LINEWIDTH)

    if SHOW_RETURN_CORRIDOR:
        add_circle(ax, rE_pos, cfg.rp_min, RETURN_COLOR,
                   label=f"Return corridor min r={cfg.rp_min:.4f}",
                   fill=False, linestyle="--", linewidth=OVERLAY_LINEWIDTH)
        add_circle(ax, rE_pos, cfg.rp_max, RETURN_COLOR,
                   label=f"Return corridor max r={cfg.rp_max:.4f}",
                   fill=False, linestyle="-.", linewidth=OVERLAY_LINEWIDTH)

    if SHOW_INVALID_BOUNDARIES and phase == "pre" and invalid_enabled:
        arm_rE = float(getattr(cfg, "ballistic_invalid_return_arm_rE", np.nan))
        moon_far_rM = float(getattr(cfg, "ballistic_invalid_return_moon_far_rM", np.nan))

        if np.isfinite(arm_rE):
            add_circle(ax, rE_pos, arm_rE, INVALID_COLOR,
                       label=f"Invalid armed rE={arm_rE:.4f}",
                       fill=False, linestyle="--", linewidth=OVERLAY_LINEWIDTH)

        if np.isfinite(moon_far_rM):
            add_circle(ax, rM_pos, moon_far_rM, INVALID_COLOR,
                       label=f"Moon-far bound rM={moon_far_rM:.4f}",
                       fill=False, linestyle="-.", linewidth=OVERLAY_LINEWIDTH)

    if SHOW_ESCAPE_BOUNDARY:
        add_circle(ax, (0.0, 0.0), cfg.r_escape, ESCAPE_COLOR,
                   label=f"Escape r={cfg.r_escape:.2f}",
                   fill=False, linestyle="--", linewidth=OVERLAY_LINEWIDTH)


def make_title(profile_key, stage_idx, phase, invalid_enabled):
    profile_title = "PPO-TLI" if profile_key == "PPO_TLI" else "PPO-MCC"
    stage_title = f"Stage {stage_idx + 1}"

    if phase == "pre" and invalid_enabled:
        main = "Pre-flyby reward landscape with invalid-return slice"
    elif phase == "pre":
        main = "Pre-flyby reward landscape"
    else:
        main = "Post-flyby return-corridor reward landscape"

    return f"{main}\n{profile_title} | {stage_title}"


def plot_heatmap(X, Y, Z, cfg, rE_pos, rM_pos, profile_key, stage_idx, phase, invalid_enabled, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)

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
        norm=TwoSlopeNorm(
            vmin=vmin,
            vcenter=0.0,
            vmax=vmax,
        ),
    )

    Z_contour = np.array(Z, copy=True)

    rE = np.sqrt((X - rE_pos[0])**2 + (Y - rE_pos[1])**2)
    rM = np.sqrt((X - rM_pos[0])**2 + (Y - rM_pos[1])**2)

    Z_contour[rE <= cfg.r_earth_impact] = np.nan
    Z_contour[rM <= cfg.r_moon_impact] = np.nan

    if SHOW_CONTOURS:
        levels = [lvl for lvl in CONTOUR_LEVELS if vmin < lvl < vmax]
        if levels:
            cs = ax.contour(X, Y, Z_contour, levels=levels, colors="k",
                            linewidths=CONTOUR_LINEWIDTH, alpha=CONTOUR_ALPHA)
            texts = ax.clabel(cs, inline=True, fontsize=max(7, TICK_SIZE - 3), fmt="%.0f")

            if FILTER_CONTOUR_LABELS_NEAR_EARTH:
                for txt in texts:
                    x_txt, y_txt = txt.get_position()
                    dE = np.linalg.norm(np.array([x_txt, y_txt]) - np.asarray(rE_pos[:2]))
                    if dE <= CONTOUR_LABEL_EARTH_CLEARANCE:
                        txt.remove()

    if SHOW_ZERO_CONTOUR and vmin < 0.0 < vmax:
        ax.contour(X, Y, Z_contour, levels=[0.0], colors="black",
                   linewidths=ZERO_CONTOUR_LINEWIDTH, alpha=0.88)

    add_overlays(ax, cfg, rE_pos, rM_pos, phase, invalid_enabled)

    ax.set_title(make_title(profile_key, stage_idx, phase, invalid_enabled), fontsize=TITLE_SIZE)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    if SHOW_GRID:
        ax.grid(True, alpha=0.25)
    else:
        ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.unique(np.array([
        vmin,
        0.75 * vmin,
        0.5 * vmin,
        0.25 * vmin,
        0.0,
        0.5 * vmax,
        vmax,
    ]))

    cbar.set_ticks(ticks)
    cbar.set_label(COLORBAR_LABEL, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    if SHOW_LEGEND:
        ax.legend(loc="upper left", fontsize=LEGEND_SIZE, framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_PNG:
        fig.savefig(out_path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_stage_summary(stage_dir, profile_key, stage_idx, stage, cfg, reward_model):
    if not SAVE_STAGE_SUMMARY_TXT:
        return

    lines = [
        "REWARD HEATMAP STAGE SUMMARY",
        "=" * 72,
        f"profile: {profile_key}",
        f"stage_index: {stage_idx}",
        f"stage_name: {stage.name}",
        "",
        "GEOMETRY",
        f"mu: {cfg.mu}",
        f"r_earth_impact: {cfg.r_earth_impact}",
        f"r_moon_impact: {cfg.r_moon_impact}",
        f"r0_earth: {cfg.r0_earth}",
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
        "",
        "SHAPING",
        f"r0_distance_flyby: {reward_model.cfg.r0_distance_flyby}",
        f"beta_distance_flyby: {reward_model.cfg.beta_distance_flyby}",
        f"flyby_reward_gate: {reward_model.cfg.flyby_reward_gate}",
        f"r0_distance_return: {reward_model.cfg.r0_distance_return}",
        f"beta_distance_return: {reward_model.cfg.beta_distance_return}",
        "",
        "INVALID DIAGNOSTIC SLICE",
        "This is a 2D projection, not the full 4D/history-dependent reward.",
        "Assumptions: closest-lunar-approach point, pre-flyby state, vrE = 0.",
        f"ballistic_invalid_preflyby_return_enabled: {getattr(cfg, 'ballistic_invalid_preflyby_return_enabled', None)}",
        f"ballistic_invalid_return_arm_rE: {getattr(cfg, 'ballistic_invalid_return_arm_rE', None)}",
        f"ballistic_invalid_return_moon_far_rM: {getattr(cfg, 'ballistic_invalid_return_moon_far_rM', None)}",
        f"ballistic_invalid_return_vrE_threshold: {getattr(cfg, 'ballistic_invalid_return_vrE_threshold', None)}",
    ]
    (stage_dir / "stage_reward_heatmap_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def generate_for_stage(profile_key, stage_idx, stage):
    cfg, _, reward_model = build_stage_objects(profile_key, stage)
    stage_dir = OUT_DIR / profile_key / f"stage_{stage_idx:02d}_{safe_stem(stage.name)}"
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
        if np.isnan(Z).any() or np.isinf(Z).any():
            print(f"[WARN] Non-finite field in {profile_key} stage {stage_idx} {name}; sanitizing.")
            Z = sanitize_field(Z)

        plot_heatmap(X, Y, Z, cfg, rE_pos, rM_pos,
                profile_key, stage_idx, phase, invalid_enabled, stage_dir / name)
        print(f"Saved {profile_key} stage {stage_idx:02d}: {name}")


def generate_profile(profile_key):
    curriculum, _, _, _ = load_profile(profile_key)
    stage_indices = list(range(len(curriculum))) if STAGE_FILTER is None else [
        i for i in STAGE_FILTER if 0 <= i < len(curriculum)
    ]
    for i in stage_indices:
        generate_for_stage(profile_key, i, curriculum[i])


def main():
    print("=" * 80)
    print("BATCH REWARD HEATMAP GENERATION")
    print("=" * 80)
    print(f"Output folder: {OUT_DIR.resolve()}")
    print(f"Grid: {GRID_NX} x {GRID_NY}")
    print("=" * 80)

    if GENERATE_PPO_TLI:
        print("\nGenerating PPO_TLI maps...")
        generate_profile("PPO_TLI")

    if GENERATE_PPO_MCC:
        print("\nGenerating PPO_MCC maps...")
        generate_profile("PPO_MCC")

    print("\nDone.")
    print(f"Saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
