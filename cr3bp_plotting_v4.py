"""
============================================================
CR3BP PLOTTING, REPORTING, AND DIAGNOSTIC UTILITIES
============================================================

This module contains the plotting and reporting tools used by
the Earth-Moon CR3BP reinforcement learning project.

It is responsible for turning rollout and evaluation data into:
- trajectory figures
- MCC debug overlays
- Earth-centered inertial views
- evaluation-batch summary figures
- training-progress plots
- reward audit outputs
- episode text and JSON reports

------------------------------------------------------------
WHAT THIS SCRIPT DOES
------------------------------------------------------------

This script provides utilities for:

1) TRAJECTORY VISUALIZATION
- rotating-frame trajectory plots
- Earth-centered inertial trajectory plots
- ballistic-reference overlays
- terminal markers for ballistic and controlled trajectories
- burn-event annotations

2) MCC / DEBUG VISUALIZATION
- separate MCC debug plots
- one ballistic overlay per MCC burn
- overlay summaries for corridor distance, success, and termination

3) EVALUATION VISUALIZATION
- full evaluation-batch trajectory grids
- spawn-theta sweep plots
- reward and geometry summaries for selected episodes

4) TRAINING DIAGNOSTICS
- mean eval reward curves
- free-return transition rate curves
- PPO metric history plots

5) REPORTING / AUDIT TOOLS
- reward-record consistency checks
- text-based episode reports
- JSON episode reports
- saved theta-sweep summaries

------------------------------------------------------------
PROJECT ROLE
------------------------------------------------------------

This module is kept separate from:
- environment physics
- reward-function logic
- PPO training logic

Its role is only to visualize, summarize, and export data that
was produced elsewhere in the project.

This keeps the project modular and allows:
- headless training with the Agg backend
- easier debugging of trajectories and rewards
- reuse in analysis scripts and manual tools

------------------------------------------------------------
HOW IT FITS INTO THE PROJECT
------------------------------------------------------------

Used by the training pipeline to:
- save evaluation plots during training
- export episode reports for selected rollouts
- generate final training-summary curves
- visualize PPO-A and PPO-B behavior in consistent formats

Used by manual / analysis workflows to:
- inspect individual trajectories
- compare ballistic vs controlled paths
- audit reward consistency step by step

------------------------------------------------------------
SUPPORTED MISSION CONTEXT
------------------------------------------------------------

The plotting tools support both main mission settings:

PPO-A:
- TLI optimization from LEO
- ballistic-reference and controlled trajectory comparison
- optional spawn-theta sweep visualization

PPO-B:
- MCC optimization from post-TLI handoff states
- MCC ballistic overlay analysis
- preservation / degradation / rescue diagnostics

------------------------------------------------------------
CHANGES RELATIVE TO A SIMPLE BASELINE PLOTTING SCRIPT
------------------------------------------------------------

(Changed from baseline: includes ballistic-reference overlays and
explicit terminal markers for both reference and controlled paths.)

(Changed from baseline: includes dedicated MCC debug plotting with
one ballistic branch per MCC burn.)

(Changed from baseline: includes reward-record auditing and detailed
episode report export in both text and JSON form.)

(Changed from baseline: includes training-level diagnostics such as
mean eval reward curves, free-return transition rates, and PPO metric
history plots.)

============================================================
"""


from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import numpy as np
import matplotlib

_MPL_BACKEND = os.environ.get("CR3BP_MPL_BACKEND", "Agg")
matplotlib.use(_MPL_BACKEND)
import matplotlib.pyplot as plt

from cr3bp_env_v4 import earth_moon_positions


def _valid_terminal_marker(marker_xy: np.ndarray, r_max: float = 1.5) -> bool:
    if marker_xy is None:
        return False
    marker_xy = np.asarray(marker_xy, dtype=np.float64).reshape(-1)
    if marker_xy.size < 2:
        return False
    r = np.linalg.norm(marker_xy[:2])
    return np.isfinite(r) and (r <= float(r_max))

def _clip_xy_path_to_radius(path_xy: np.ndarray, r_max: float = 1.5) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)

    if path_xy.ndim != 2 or path_xy.shape[1] < 2 or len(path_xy) == 0:
        return path_xy

    r = np.linalg.norm(path_xy[:, :2], axis=1)
    inside = r <= float(r_max)

    if np.all(inside):
        return path_xy

    if not np.any(inside):
        return path_xy[:0]

    first_outside = np.argmax(~inside)
    if inside[first_outside]:
        return path_xy

    keep_n = max(2, first_outside)
    return path_xy[:keep_n]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(obj):
    """
    Convert nested numpy-heavy structures into JSON-safe Python objects.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if np.isnan(x):
            return None
        if np.isposinf(x):
            return "inf"
        if np.isneginf(x):
            return "-inf"
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isposinf(obj):
            return "inf"
        if np.isneginf(obj):
            return "-inf"
        return obj
    return obj


def _stack_or_empty(rows, width=None, dtype=np.float64):
    """
    Stack a list of 1D arrays into shape (N, D). If empty, return (0, D).
    """
    if rows is None or len(rows) == 0:
        if width is None:
            return np.zeros((0, 0), dtype=dtype)
        return np.zeros((0, width), dtype=dtype)

    arrs = [np.asarray(r, dtype=dtype).reshape(-1) for r in rows]
    if width is None:
        width = max(a.size for a in arrs)

    out = np.full((len(arrs), width), np.nan, dtype=dtype)
    for i, a in enumerate(arrs):
        n = min(width, a.size)
        out[i, :n] = a[:n]
    return out


def _extract_action_history_arrays(action_history: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert env.action_history into compact numeric arrays for .npz export.
    """
    if action_history is None:
        action_history = []

    state_before = []
    state_after = []
    obs_before = []
    obs_after = []

    step_idx = []
    time_before = []
    time_after = []
    ax_raw = []
    ay_raw = []
    tau_raw = []
    tau_true_if_tli = []
    u01_raw = []
    u01_exec = []
    dv_mag = []
    dt_effective = []
    reward = []
    terminated = []
    truncated = []

    rE = []
    rM = []
    dv_used = []
    flyby_done = []
    corridor_hit = []
    ballistic_hit = []
    left_leo = []

    burn_kind_codes = []
    burn_kind_map: Dict[str, int] = {}
    next_code = 0

    for row in action_history:
        state_before.append(row.get("state_before_action", []))
        state_after.append(row.get("state_after_step", []))
        obs_before.append(row.get("obs_before_action", []))
        obs_after.append(row.get("obs_after_step", []))

        step_idx.append(float(row.get("step_idx", np.nan)))
        time_before.append(float(row.get("time_before", np.nan)))
        time_after.append(float(row.get("time_after", np.nan)))
        ax_raw.append(float(row.get("ax_raw", np.nan)))
        ay_raw.append(float(row.get("ay_raw", np.nan)))
        tau_raw.append(float(row.get("tau_raw", np.nan)))
        tau_true_if_tli.append(float(row.get("tau_true_if_tli", np.nan)))
        u01_raw.append(float(row.get("u01_raw", np.nan)))
        u01_exec.append(float(row.get("u01_exec", np.nan)))
        dv_mag.append(float(row.get("dv_mag", np.nan)))
        dt_effective.append(float(row.get("dt_effective", np.nan)))
        reward.append(float(row.get("reward", np.nan)))
        terminated.append(1.0 if bool(row.get("terminated", False)) else 0.0)
        truncated.append(1.0 if bool(row.get("truncated", False)) else 0.0)

        info_sel = row.get("info_selected", {}) or {}
        rE.append(float(info_sel.get("rE", np.nan)))
        rM.append(float(info_sel.get("rM", np.nan)))
        dv_used.append(float(info_sel.get("dv_used", np.nan)))
        flyby_done.append(1.0 if bool(info_sel.get("flyby_done", False)) else 0.0)
        corridor_hit.append(1.0 if bool(info_sel.get("return_corridor_hit_postflyby", False)) else 0.0)
        ballistic_hit.append(1.0 if bool(info_sel.get("ballistic_tli_corridor_hit", False)) else 0.0)
        left_leo.append(1.0 if bool(info_sel.get("left_leo", False)) else 0.0)

        kind = str(row.get("burn_kind", "UNKNOWN"))
        if kind not in burn_kind_map:
            burn_kind_map[kind] = next_code
            next_code += 1
        burn_kind_codes.append(float(burn_kind_map[kind]))

    obs_width = 0
    if len(obs_before) > 0:
        obs_width = max(len(np.asarray(x).reshape(-1)) for x in obs_before)

    arrays = {
        "step_state_before": _stack_or_empty(state_before, width=4, dtype=np.float64),
        "step_state_after": _stack_or_empty(state_after, width=4, dtype=np.float64),
        "step_obs_before": _stack_or_empty(obs_before, width=obs_width, dtype=np.float32),
        "step_obs_after": _stack_or_empty(obs_after, width=obs_width, dtype=np.float32),

        "step_idx": np.asarray(step_idx, dtype=np.float64),
        "step_time_before": np.asarray(time_before, dtype=np.float64),
        "step_time_after": np.asarray(time_after, dtype=np.float64),
        "step_ax_raw": np.asarray(ax_raw, dtype=np.float64),
        "step_ay_raw": np.asarray(ay_raw, dtype=np.float64),
        "step_tau_raw": np.asarray(tau_raw, dtype=np.float64),
        "step_tau_true_if_tli": np.asarray(tau_true_if_tli, dtype=np.float64),
        "step_u01_raw": np.asarray(u01_raw, dtype=np.float64),
        "step_u01_exec": np.asarray(u01_exec, dtype=np.float64),
        "step_dv_mag": np.asarray(dv_mag, dtype=np.float64),
        "step_dt_effective": np.asarray(dt_effective, dtype=np.float64),
        "step_reward": np.asarray(reward, dtype=np.float64),
        "step_terminated": np.asarray(terminated, dtype=np.float64),
        "step_truncated": np.asarray(truncated, dtype=np.float64),

        "step_info_rE": np.asarray(rE, dtype=np.float64),
        "step_info_rM": np.asarray(rM, dtype=np.float64),
        "step_info_dv_used": np.asarray(dv_used, dtype=np.float64),
        "step_info_flyby_done": np.asarray(flyby_done, dtype=np.float64),
        "step_info_corridor_hit": np.asarray(corridor_hit, dtype=np.float64),
        "step_info_ballistic_hit": np.asarray(ballistic_hit, dtype=np.float64),
        "step_info_left_leo": np.asarray(left_leo, dtype=np.float64),

        "step_burn_kind_code": np.asarray(burn_kind_codes, dtype=np.float64),
    }

    burn_kind_lookup = {int(v): k for k, v in burn_kind_map.items()}
    return arrays, burn_kind_lookup


def save_eval_episode_archive_npz_json(
    ep: Dict[str, Any],
    out_dir: Path,
    stem: str,
    clip_radius: float = 1.5,
) -> Dict[str, Path]:
    """
    Save one evaluation episode in a reusable analysis format:
    - .npz for arrays
    - .json for metadata / nested structures
    """
    out_dir = _ensure_dir(Path(out_dir))

    traj = np.asarray(ep.get("traj", np.zeros((0, 4))), dtype=np.float64)
    t_hist = np.asarray(ep.get("t_hist", np.zeros((0,))), dtype=np.float64)

    ballistic_ref_traj_raw = ep.get("ballistic_ref_traj", None)
    ballistic_ref_t_hist_raw = ep.get("ballistic_ref_t_hist", None)

    ballistic_ref_traj = (
        np.asarray(ballistic_ref_traj_raw, dtype=np.float64)
        if ballistic_ref_traj_raw is not None
        else np.zeros((0, 4), dtype=np.float64)
    )
    ballistic_ref_t_hist = (
        np.asarray(ballistic_ref_t_hist_raw, dtype=np.float64)
        if ballistic_ref_t_hist_raw is not None
        else np.zeros((0,), dtype=np.float64)
    )

    traj_rot_clip15_xy = _clip_xy_path_to_radius(traj[:, :2], r_max=clip_radius) if traj.ndim == 2 and traj.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float64)
    ballistic_ref_rot_clip15_xy = _clip_xy_path_to_radius(ballistic_ref_traj[:, :2], r_max=clip_radius) if ballistic_ref_traj.ndim == 2 and ballistic_ref_traj.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float64)

    terminal_marker_rot = np.asarray(ep.get("terminal_marker_rot", np.zeros((0,))), dtype=np.float64)
    ballistic_terminal_marker_rot = np.asarray(ep.get("ballistic_terminal_marker_rot", np.zeros((0,))), dtype=np.float64)

    burn_events = ep.get("burn_events", []) or []
    burn_pos = []
    burn_dv = []
    burn_mag = []
    burn_tau = []
    burn_ax = []
    burn_ay = []

    for ev in burn_events:
        burn_pos.append(np.asarray(ev.get("pos_rot", []), dtype=np.float64).reshape(-1))
        burn_dv.append(np.asarray(ev.get("dv_vec_rot", []), dtype=np.float64).reshape(-1))
        burn_mag.append(float(ev.get("dv_mag", np.nan)))
        burn_tau.append(float(ev.get("tau_raw", np.nan)))
        burn_ax.append(float(ev.get("ax_raw", np.nan)))
        burn_ay.append(float(ev.get("ay_raw", np.nan)))

    burn_pos_arr = _stack_or_empty(burn_pos, width=2, dtype=np.float64)
    burn_dv_arr = _stack_or_empty(burn_dv, width=2, dtype=np.float64)
    burn_mag_arr = np.asarray(burn_mag, dtype=np.float64)
    burn_tau_arr = np.asarray(burn_tau, dtype=np.float64)
    burn_ax_arr = np.asarray(burn_ax, dtype=np.float64)
    burn_ay_arr = np.asarray(burn_ay, dtype=np.float64)

    action_history = ep.get("action_history", []) or []
    action_arrays, burn_kind_lookup = _extract_action_history_arrays(action_history)

    overlay_count = len(ep.get("mcc_ballistic_overlays", []) or [])
    overlay_lengths = []
    for ov in (ep.get("mcc_ballistic_overlays", []) or []):
        traj_ov = np.asarray(ov.get("traj", []), dtype=np.float64)
        overlay_lengths.append(int(traj_ov.shape[0]) if traj_ov.ndim == 2 else 0)

    npz_path = out_dir / f"{stem}_arrays.npz"
    json_path = out_dir / f"{stem}_meta.json"

    save_dict = {
        "traj_rot_full": traj,
        "traj_rot_clip15_xy": traj_rot_clip15_xy,
        "t_hist": t_hist,
        "ballistic_ref_rot_full": ballistic_ref_traj,
        "ballistic_ref_rot_clip15_xy": ballistic_ref_rot_clip15_xy,
        "ballistic_ref_t_hist": ballistic_ref_t_hist,
        "terminal_marker_rot": terminal_marker_rot,
        "ballistic_terminal_marker_rot": ballistic_terminal_marker_rot,
        "burn_pos_rot": burn_pos_arr,
        "burn_dv_vec_rot": burn_dv_arr,
        "burn_dv_mag": burn_mag_arr,
        "burn_tau_raw": burn_tau_arr,
        "burn_ax_raw": burn_ax_arr,
        "burn_ay_raw": burn_ay_arr,
        "overlay_count": np.asarray([overlay_count], dtype=np.int64),
        "overlay_lengths": np.asarray(overlay_lengths, dtype=np.int64),
    }
    save_dict.update(action_arrays)

    np.savez_compressed(npz_path, **save_dict)

    meta = {
        "format_version": 1,
        "clip_radius": float(clip_radius),
        "coordinate_frame": "rotating_cr3bp_nondim",
        "state_schema": ep.get("state_schema", ["x", "y", "vx", "vy"]),
        "obs_schema": ep.get("obs_schema", []),
        "action_schema": ep.get("action_schema", ["ax_raw", "ay_raw", "tau_raw"]),
        "obs_dim": int(ep.get("obs_dim", 0)),
        "state_dim": int(ep.get("state_dim", 4)),
        "reason": str(ep.get("reason", "")),
        "reward_sum": float(ep.get("reward_sum", np.nan)),
        "dv_used": float(ep.get("dv_used", np.nan)),
        "dv0": float(ep.get("dv0", np.nan)),
        "min_rM": float(ep.get("min_rM", np.nan)),
        "min_rE": float(ep.get("min_rE", np.nan)),
        "min_rE_postflyby": float(ep.get("min_rE_postflyby", np.nan)),
        "moon_corridor_miss": float(ep.get("moon_corridor_miss", np.nan)),
        "return_corridor_miss": float(ep.get("return_corridor_miss", np.nan)),
        "vrel_at_min_rM": float(ep.get("vrel_at_min_rM", np.nan)),
        "flyby_done": bool(ep.get("flyby_done", False)),
        "corridor_hit": bool(ep.get("corridor_hit", False)),
        "ballistic_success": bool(ep.get("ballistic_success", False)),
        "trajectory_success": bool(ep.get("trajectory_success", False)),
        "success_flag_latched": bool(ep.get("success_flag_latched", False)),
        "left_leo": bool(ep.get("left_leo", False)),
        "left_leo_step": float(ep.get("left_leo_step", np.nan)),
        "tli_tau": float(ep.get("tli_tau", np.nan)),
        "tli_ax": float(ep.get("tli_ax", np.nan)),
        "tli_ay": float(ep.get("tli_ay", np.nan)),
        "ballistic_tli_reward": float(ep.get("ballistic_tli_reward", np.nan)),
        "ballistic_tli_min_rM": float(ep.get("ballistic_tli_min_rM", np.nan)),
        "ballistic_tli_corridor_dist": float(ep.get("ballistic_tli_corridor_dist", np.nan)),
        "ballistic_tli_corridor_hit": bool(ep.get("ballistic_tli_corridor_hit", False)),
        "burn_kind_lookup": burn_kind_lookup,
        "burn_events": _to_jsonable(burn_events),
        "mcc_ballistic_overlays": _to_jsonable(ep.get("mcc_ballistic_overlays", [])),
        "reset_debug": _to_jsonable(ep.get("reset_debug", {})),
        "info_last": _to_jsonable(ep.get("info_last", {})),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "npz_path": npz_path,
        "json_path": json_path,
    }


def save_training_history_npz(
    eval_history: List[Dict[str, Any]],
    ppo_history: List[Dict[str, Any]],
    out_dir: Path,
    stem: str = "final_training_curves",
) -> Path:
    """
    Save the raw vectors behind the final training plots.
    """
    out_dir = _ensure_dir(Path(out_dir))
    out_path = out_dir / f"{stem}.npz"

    eval_steps = np.asarray([row.get("step", np.nan) for row in eval_history], dtype=np.float64)
    eval_reward_mean = np.asarray([row.get("reward_mean", np.nan) for row in eval_history], dtype=np.float64)
    eval_dv_mean = np.asarray([row.get("dv_mean", np.nan) for row in eval_history], dtype=np.float64)
    eval_dv_std = np.asarray([row.get("dv_std", np.nan) for row in eval_history], dtype=np.float64)
    eval_preservation_rate = np.asarray([row.get("preservation_rate", np.nan) for row in eval_history], dtype=np.float64)
    eval_degradation_rate = np.asarray([row.get("degradation_rate", np.nan) for row in eval_history], dtype=np.float64)
    eval_rescue_rate = np.asarray([row.get("rescue_rate", np.nan) for row in eval_history], dtype=np.float64)
    eval_unchanged_bad_rate = np.asarray([row.get("unchanged_bad_rate", np.nan) for row in eval_history], dtype=np.float64)

    # New success logging arrays
    eval_success_rate = np.asarray([row.get("success_rate", np.nan) for row in eval_history], dtype=np.float64)
    eval_ballistic_success_rate = np.asarray([row.get("ballistic_success_rate", np.nan) for row in eval_history], dtype=np.float64)
    eval_trajectory_success_rate = np.asarray([row.get("trajectory_success_rate", np.nan) for row in eval_history], dtype=np.float64)

    eval_success_count = np.asarray([row.get("success_count", np.nan) for row in eval_history], dtype=np.float64)
    eval_ballistic_success_count = np.asarray([row.get("ballistic_success_count", np.nan) for row in eval_history], dtype=np.float64)
    eval_trajectory_success_count = np.asarray([row.get("trajectory_success_count", np.nan) for row in eval_history], dtype=np.float64)
    eval_n_episodes = np.asarray([row.get("n_eval_episodes", np.nan) for row in eval_history], dtype=np.float64)

    ppo_steps = np.asarray([row.get("step", np.nan) for row in ppo_history], dtype=np.float64)

    def ppo_arr(key: str) -> np.ndarray:
        return np.asarray([row.get(key, np.nan) for row in ppo_history], dtype=np.float64)

    np.savez_compressed(
        out_path,
        eval_step=eval_steps,
        eval_reward_mean=eval_reward_mean,
        eval_preservation_rate=eval_preservation_rate,
        eval_degradation_rate=eval_degradation_rate,
        eval_rescue_rate=eval_rescue_rate,
        eval_unchanged_bad_rate=eval_unchanged_bad_rate,
        eval_dv_mean=eval_dv_mean,
        eval_dv_std=eval_dv_std,

        # New success logging arrays
        eval_success_rate=eval_success_rate,
        eval_ballistic_success_rate=eval_ballistic_success_rate,
        eval_trajectory_success_rate=eval_trajectory_success_rate,
        eval_success_count=eval_success_count,
        eval_ballistic_success_count=eval_ballistic_success_count,
        eval_trajectory_success_count=eval_trajectory_success_count,
        eval_n_episodes=eval_n_episodes,

        ppo_step=ppo_steps,
        approx_kl=ppo_arr("approx_kl"),
        clip_fraction=ppo_arr("clip_fraction"),
        clip_range=ppo_arr("clip_range"),
        policy_gradient_loss=ppo_arr("policy_gradient_loss"),
        value_loss=ppo_arr("value_loss"),
        loss=ppo_arr("loss"),
        entropy_loss=ppo_arr("entropy_loss"),
        explained_variance=ppo_arr("explained_variance"),
        std=ppo_arr("std"),
        learning_rate=ppo_arr("learning_rate"),
    )

    return out_path

def classify_free_return_outcome(initial_free_return: bool, final_free_return: bool) -> str:
    """
    Episode-level free-return transition classification.

    preserved    : started good, ended good
    degraded     : started good, ended bad
    rescued      : started bad, ended good
    unchanged_bad: started bad, ended bad
    """
    initial_free_return = bool(initial_free_return)
    final_free_return = bool(final_free_return)

    if initial_free_return and final_free_return:
        return "preserved"
    if initial_free_return and (not final_free_return):
        return "degraded"
    if (not initial_free_return) and final_free_return:
        return "rescued"
    return "unchanged_bad"



# ============================================================
# 4) PLOTTING 
# ============================================================



def plot_trajectory(
    cfg,
    traj,
    burns=None,
    burn_events=None,
    ballistic_ref_traj=None,
    ballistic_terminal_marker=None,
    terminal_marker=None,
    title="",
    out_path=None,
):
    mu = cfg.mu
    rE_pos, rM_pos = earth_moon_positions(mu)

    traj = np.asarray(traj, dtype=np.float64)
    xs = traj[:, 0]
    ys = traj[:, 1]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Main trajectory
    ax.plot(xs, ys, linewidth=1.2, label="Trajectory")

    # Ballistic reference after committed TLI
    if ballistic_ref_traj is not None and len(ballistic_ref_traj) > 1:
        ballistic_ref_traj = np.asarray(ballistic_ref_traj, dtype=np.float64)

        max_radius = 1.5
        r_norm = np.linalg.norm(ballistic_ref_traj[:, :2], axis=1)

        inside = r_norm <= max_radius
        if np.any(~inside):
            cutoff = np.argmax(~inside)
            ballistic_ref_traj = ballistic_ref_traj[:max(2, cutoff)]

        if len(ballistic_ref_traj) > 1:
            ax.plot(
                ballistic_ref_traj[:, 0],
                ballistic_ref_traj[:, 1],
                linestyle=":",
                linewidth=1.6,
                color="orange",
                label="Ballistic after TLI commit",
            )
    
    # Ballistic terminal marker (orange X)
    if _valid_terminal_marker(ballistic_terminal_marker, r_max=1.5):
        p = np.asarray(ballistic_terminal_marker, dtype=np.float64).reshape(-1)
        ax.scatter(
            p[0], p[1],
            s=90,
            marker="x",
            color="orange",
            linewidths=2.2,
            label="Ballistic termination",
            zorder=8,
        )

    # Real episode terminal marker (red X)
    if _valid_terminal_marker(terminal_marker, r_max=1.5):
        p = np.asarray(terminal_marker, dtype=np.float64).reshape(-1)
        ax.scatter(
            p[0], p[1],
            s=100,
            marker="x",
            color="red",
            linewidths=2.4,
            label="Episode termination",
            zorder=9,
        )

    # Primaries
    earth_disk = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.r_earth_impact,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        label=f"Earth physical radius r={cfg.r_earth_impact:.4f}",
        zorder=5,
    )
    ax.add_patch(earth_disk)

    moon_disk = plt.Circle(
        (rM_pos[0], rM_pos[1]),
        cfg.r_moon_impact,
        facecolor="gray",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        label=f"Moon physical radius r={cfg.r_moon_impact:.4f}",
        zorder=5,
    )
    ax.add_patch(moon_disk)

    leo_start_orbit = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.r0_earth,
        fill=False,
        linestyle=":",
        linewidth=1.2,
        edgecolor="lightblue",
        alpha=0.8,
        label=f"LEO start orbit r={cfg.r0_earth:.4f}",
    )
    ax.add_patch(leo_start_orbit)

    earth_corridor_min = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.rp_min,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
        label=f"Earth corridor min r={cfg.rp_min:.4f}",
    )
    ax.add_patch(earth_corridor_min)

    earth_corridor_max = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.rp_max,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
        label=f"Earth corridor max r={cfg.rp_max:.4f}",
    )
    ax.add_patch(earth_corridor_max)

    moon_flyby_upper = plt.Circle(
        (rM_pos[0], rM_pos[1]),
        cfg.r_moon_flyby,
        fill=False,
        linestyle=":",
        linewidth=1.5,
        label=f"Moon flyby bound r={cfg.r_moon_flyby:.4f}",
    )
    ax.add_patch(moon_flyby_upper)

    # Burn vectors
    summary_lines = []
    if burn_events is not None and len(burn_events) > 0:
        for ev in burn_events:
            pos_b = np.asarray(ev["pos_rot"], dtype=np.float64)
            dv_b = np.asarray(ev["dv_vec_rot"], dtype=np.float64)
            mag_b = float(ev.get("dv_mag", 0.0))
            kind = str(ev.get("kind", "Burn"))

            if mag_b <= 0.0:
                continue

            scale = 0.06 / max(mag_b, 1e-12)
            vec_plot = dv_b * scale

            ax.scatter(pos_b[0], pos_b[1], s=30, marker="x", label=kind)
            ax.arrow(
                pos_b[0],
                pos_b[1],
                vec_plot[0],
                vec_plot[1],
                head_width=0.01,
                head_length=0.015,
                length_includes_head=True,
                alpha=0.9,
            )
            ax.text(pos_b[0], pos_b[1], kind, fontsize=8)

            tau_raw = float(ev.get("tau_raw", np.nan))
            summary_lines.append(
                f"{kind}: dv={mag_b:.4f}, ax={ev.get('ax_raw', np.nan):.3f}, "
                f"ay={ev.get('ay_raw', np.nan):.3f}, tau_raw={tau_raw:.3f}"
            )

    if len(summary_lines) > 0:
        text_block = "\n".join(summary_lines[:8])
        fig.text(
            0.02, 0.02, text_block,
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (rotating frame, nondim)")
    ax.set_ylabel("y (rotating frame, nondim)")
    ax.set_title(title if title else "CR3BP trajectory")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    ax.legend(uniq_handles, uniq_labels, loc="best")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if out_path is None:
        out_path = "traj.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory_mcc_debug(
    cfg,
    traj,
    burn_events=None,
    ballistic_ref_traj=None,
    ballistic_terminal_marker=None,
    terminal_marker=None,
    mcc_ballistic_overlays=None,
    title="",
    out_path=None,
):
    """
    Separate MCC debug figure.

    This keeps the normal CR3BP plot untouched and creates a second plot
    showing:
    - real flown trajectory
    - original ballistic after TLI commit
    - one ballistic coast-only overlay for each MCC burn
    - matching termination markers
    """
    mu = cfg.mu
    rE_pos, rM_pos = earth_moon_positions(mu)

    traj = np.asarray(traj, dtype=np.float64)
    xs = traj[:, 0]
    ys = traj[:, 1]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # --------------------------------------------------------
    # Main real trajectory
    # --------------------------------------------------------
    ax.plot(xs, ys, linewidth=1.4, label="Real trajectory")

    # --------------------------------------------------------
    # Original ballistic reference after committed TLI
    # --------------------------------------------------------
    if ballistic_ref_traj is not None and len(ballistic_ref_traj) > 1:
        ballistic_ref_traj = np.asarray(ballistic_ref_traj, dtype=np.float64)
        ballistic_xy = _clip_xy_path_to_radius(ballistic_ref_traj[:, :2], r_max=1.5)

        if len(ballistic_xy) > 1:
            ax.plot(
                ballistic_xy[:, 0],
                ballistic_xy[:, 1],
                linestyle=":",
                linewidth=1.8,
                color="orange",
                label="Ballistic after TLI commit",
            )

    if _valid_terminal_marker(ballistic_terminal_marker, r_max=1.5):
        p = np.asarray(ballistic_terminal_marker, dtype=np.float64).reshape(-1)
        ax.scatter(
            p[0], p[1],
            s=95,
            marker="x",
            color="orange",
            linewidths=2.2,
            label="Ballistic TLI termination",
            zorder=8,
        )

    # --------------------------------------------------------
    # MCC ballistic overlay branches
    # --------------------------------------------------------
    overlay_summary_lines = []

    if mcc_ballistic_overlays is not None and len(mcc_ballistic_overlays) > 0:
        cmap = plt.cm.get_cmap("tab10", max(1, len(mcc_ballistic_overlays)))

        for i, overlay in enumerate(mcc_ballistic_overlays):
            color = cmap(i)

            overlay_traj = np.asarray(overlay.get("traj_rot", []), dtype=np.float64)
            if overlay_traj.ndim != 2 or overlay_traj.shape[0] < 2 or overlay_traj.shape[1] < 2:
                continue

            overlay_xy = _clip_xy_path_to_radius(overlay_traj[:, :2], r_max=1.5)
            if len(overlay_xy) > 1:
                ax.plot(
                    overlay_xy[:, 0],
                    overlay_xy[:, 1],
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.95,
                    label=f"MCC {i+1} ballistic",
                )

            p_term = overlay.get("terminal_marker_rot", None)
            if _valid_terminal_marker(p_term, r_max=1.5):
                p_term = np.asarray(p_term, dtype=np.float64).reshape(-1)
                ax.scatter(
                    p_term[0],
                    p_term[1],
                    s=85,
                    marker="x",
                    color=color,
                    linewidths=2.0,
                    zorder=8,
                    label=f"MCC {i+1} termination",
                )

            overlay_summary_lines.append(
                f"MCC {i+1}: "
                f"dv={float(overlay.get('dv_mag', np.nan)):.4f}, "
                f"success={bool(overlay.get('success', False))}, "
                f"term={overlay.get('term_reason', '')}, "
                f"corridor_dist={float(overlay.get('corridor_dist', np.nan)):.4f}"
            )

    # --------------------------------------------------------
    # Real episode terminal marker
    # --------------------------------------------------------
    if _valid_terminal_marker(terminal_marker, r_max=1.5):
        p = np.asarray(terminal_marker, dtype=np.float64).reshape(-1)
        ax.scatter(
            p[0], p[1],
            s=100,
            marker="x",
            color="red",
            linewidths=2.4,
            label="Real episode termination",
            zorder=9,
        )

    # --------------------------------------------------------
    # Burn markers
    # --------------------------------------------------------
    if burn_events is not None and len(burn_events) > 0:
        for ev in burn_events:
            pos_b = np.asarray(ev["pos_rot"], dtype=np.float64)
            dv_b = np.asarray(ev["dv_vec_rot"], dtype=np.float64)
            mag_b = float(ev.get("dv_mag", 0.0))
            kind = str(ev.get("kind", "Burn"))

            if mag_b <= 0.0:
                continue

            scale = 0.06 / max(mag_b, 1e-12)
            vec_plot = dv_b * scale

            ax.scatter(
                pos_b[0],
                pos_b[1],
                s=28,
                marker="o",
                alpha=0.9,
                zorder=7,
            )
            ax.arrow(
                pos_b[0],
                pos_b[1],
                vec_plot[0],
                vec_plot[1],
                head_width=0.01,
                head_length=0.015,
                length_includes_head=True,
                alpha=0.85,
            )
            ax.text(pos_b[0], pos_b[1], kind, fontsize=8)

    # --------------------------------------------------------
    # Primaries and geometry
    # --------------------------------------------------------
    earth_disk = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.r_earth_impact,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        label=f"Earth physical radius r={cfg.r_earth_impact:.4f}",
        zorder=5,
    )
    ax.add_patch(earth_disk)

    moon_disk = plt.Circle(
        (rM_pos[0], rM_pos[1]),
        cfg.r_moon_impact,
        facecolor="gray",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        label=f"Moon physical radius r={cfg.r_moon_impact:.4f}",
        zorder=5,
    )
    ax.add_patch(moon_disk)

    leo_start_orbit = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.r0_earth,
        fill=False,
        linestyle=":",
        linewidth=1.2,
        edgecolor="lightblue",
        alpha=0.8,
        label=f"LEO start orbit r={cfg.r0_earth:.4f}",
    )
    ax.add_patch(leo_start_orbit)

    earth_corridor_min = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.rp_min,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
        label=f"Earth corridor min r={cfg.rp_min:.4f}",
    )
    ax.add_patch(earth_corridor_min)

    earth_corridor_max = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.rp_max,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
        label=f"Earth corridor max r={cfg.rp_max:.4f}",
    )
    ax.add_patch(earth_corridor_max)

    moon_flyby_upper = plt.Circle(
        (rM_pos[0], rM_pos[1]),
        cfg.r_moon_flyby,
        fill=False,
        linestyle=":",
        linewidth=1.5,
        label=f"Moon flyby bound r={cfg.r_moon_flyby:.4f}",
    )
    ax.add_patch(moon_flyby_upper)

    if len(overlay_summary_lines) > 0:
        text_block = "\n".join(overlay_summary_lines[:12])
        fig.text(
            0.02, 0.02, text_block,
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (rotating frame, nondim)")
    ax.set_ylabel("y (rotating frame, nondim)")
    ax.set_title(title if title else "CR3BP MCC debug trajectory")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    ax.legend(uniq_handles, uniq_labels, loc="best")

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if out_path is None:
        out_path = "traj_mcc_debug.png"

    plt.savefig(out_path, dpi=150)
    plt.close(fig)




def plot_trajectory_earth_centered_inertial(
    cfg,
    traj,
    t_hist,
    ballistic_ref_traj=None,
    ballistic_ref_t_hist=None,
    title="",
    out_path=None,
):
    mu = float(cfg.mu)

    traj = np.asarray(traj, dtype=np.float64)
    t_hist = np.asarray(t_hist, dtype=np.float64).reshape(-1)

    N = min(traj.shape[0], t_hist.shape[0])
    if N < 2:
        return

    rR = traj[:N, 0:2]
    t = t_hist[:N]

    ct = np.cos(t)
    st = np.sin(t)

    # Spacecraft: rotating -> inertial
    xI = ct * rR[:, 0] - st * rR[:, 1]
    yI = st * rR[:, 0] + ct * rR[:, 1]
    rI = np.stack([xI, yI], axis=1)

    # Earth and Moon locations in rotating frame
    rE_R = np.array([-mu, 0.0], dtype=np.float64)
    rM_R = np.array([1.0 - mu, 0.0], dtype=np.float64)

    # Earth and Moon inertial positions on spacecraft time grid
    xE = ct * rE_R[0] - st * rE_R[1]
    yE = st * rE_R[0] + ct * rE_R[1]
    rE_I = np.stack([xE, yE], axis=1)

    xM = ct * rM_R[0] - st * rM_R[1]
    yM = st * rM_R[0] + ct * rM_R[1]
    rM_I = np.stack([xM, yM], axis=1)

    # Earth-centered inertial coordinates
    r_sc_E = rI - rE_I
    r_m_E = rM_I - rE_I

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(r_sc_E[:, 0], r_sc_E[:, 1], linewidth=1.2, label="Spacecraft")
    ax.plot(r_m_E[:, 0], r_m_E[:, 1], linewidth=1.0, alpha=0.7, label="Moon path")

    # Real-size Earth disk at origin
    earth_disk = plt.Circle(
        (0.0, 0.0),
        cfg.r_earth_impact,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        label=f"Earth physical radius r={cfg.r_earth_impact:.4f}",
        zorder=5,
    )
    ax.add_patch(earth_disk)

    # Real-size Moon disk at final Moon position
    moon_center = r_m_E[-1]
    moon_disk = plt.Circle(
        (moon_center[0], moon_center[1]),
        cfg.r_moon_impact,
        facecolor="gray",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        label=f"Moon physical radius r={cfg.r_moon_impact:.4f}",
        zorder=5,
    )
    ax.add_patch(moon_disk)

    # Optional ballistic reference overlay in inertial Earth-centered frame
    if ballistic_ref_traj is not None and ballistic_ref_t_hist is not None:
        ballistic_ref_traj = np.asarray(ballistic_ref_traj, dtype=np.float64)
        ballistic_ref_t_hist = np.asarray(ballistic_ref_t_hist, dtype=np.float64).reshape(-1)

        NB = min(ballistic_ref_traj.shape[0], ballistic_ref_t_hist.shape[0])
        if NB >= 2:
            rB_R = ballistic_ref_traj[:NB, 0:2]
            tB = ballistic_ref_t_hist[:NB]

            ctB = np.cos(tB)
            stB = np.sin(tB)

            # Ballistic: rotating -> inertial
            xB = ctB * rB_R[:, 0] - stB * rB_R[:, 1]
            yB = stB * rB_R[:, 0] + ctB * rB_R[:, 1]
            rB_I = np.stack([xB, yB], axis=1)

            # Earth on ballistic time grid
            xE_B = ctB * rE_R[0] - stB * rE_R[1]
            yE_B = stB * rE_R[0] + ctB * rE_R[1]
            rE_I_B = np.stack([xE_B, yE_B], axis=1)

            # Earth-centered ballistic path
            rB_E = rB_I - rE_I_B

            max_radius = 1.5
            r_norm = np.linalg.norm(rB_E, axis=1)
            inside = r_norm <= max_radius
            if np.any(~inside):
                cutoff = np.argmax(~inside)
                rB_E = rB_E[:max(2, cutoff)]

            if len(rB_E) > 1:
                ax.plot(
                    rB_E[:, 0],
                    rB_E[:, 1],
                    linestyle=":",
                    linewidth=1.6,
                    color="orange",
                    label="Ballistic after TLI commit",
                )

            # Optional Moon disk for ballistic timeline end
            moon_ball_center = np.array([
                np.cos(tB[-1]) * rM_R[0] - np.sin(tB[-1]) * rM_R[1],
                np.sin(tB[-1]) * rM_R[0] + np.cos(tB[-1]) * rM_R[1],
            ], dtype=np.float64)
            earth_ball_center = np.array([
                np.cos(tB[-1]) * rE_R[0] - np.sin(tB[-1]) * rE_R[1],
                np.sin(tB[-1]) * rE_R[0] + np.cos(tB[-1]) * rE_R[1],
            ], dtype=np.float64)
            moon_ball_center_E = moon_ball_center - earth_ball_center

            moon_disk_ballistic = plt.Circle(
                (moon_ball_center_E[0], moon_ball_center_E[1]),
                cfg.r_moon_impact,
                facecolor="gray",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.35,
                zorder=4,
            )
            ax.add_patch(moon_disk_ballistic)

    leo_start_orbit = plt.Circle(
        (0.0, 0.0),
        cfg.r0_earth,
        fill=False,
        linestyle=":",
        linewidth=1.2,
        edgecolor="lightblue",
        alpha=0.8,
        label=f"LEO start orbit r={cfg.r0_earth:.4f}",
    )
    ax.add_patch(leo_start_orbit)

    earth_corridor_min = plt.Circle(
        (0.0, 0.0),
        cfg.rp_min,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
        label=f"Earth corridor min r={cfg.rp_min:.4f}",
    )
    ax.add_patch(earth_corridor_min)

    earth_corridor_max = plt.Circle(
        (0.0, 0.0),
        cfg.rp_max,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
        label=f"Earth corridor max r={cfg.rp_max:.4f}",
    )
    ax.add_patch(earth_corridor_max)

    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (Earth-centered inertial, nondim)")
    ax.set_ylabel("y (Earth-centered inertial, nondim)")
    ax.set_title(title if title else "Earth-centered inertial trajectory")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    ax.legend(uniq_handles, uniq_labels, loc="best")

    plt.tight_layout()

    if out_path is None:
        out_path = "traj_inert.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_spawn_theta_sweep(
    cfg,
    sweep_results,
    title="",
    out_path=None,
):
    mu = cfg.mu
    rE_pos, rM_pos = earth_moon_positions(mu)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.gca()

    def clip_path_to_radius(path_xy: np.ndarray, r_max: float = 1.5) -> np.ndarray:
        path_xy = np.asarray(path_xy, dtype=np.float64)
        if path_xy.ndim != 2 or path_xy.shape[1] < 2 or len(path_xy) == 0:
            return path_xy

        r = np.linalg.norm(path_xy[:, :2], axis=1)
        inside = r <= float(r_max)

        if np.all(inside):
            return path_xy

        if not np.any(inside):
            return path_xy[:0]

        first_outside = np.argmax(~inside)
        if inside[first_outside]:
            return path_xy

        keep_n = max(2, first_outside)
        return path_xy[:keep_n]

    # Primaries
    earth_disk = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.r_earth_impact,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(earth_disk)

    moon_disk = plt.Circle(
        (rM_pos[0], rM_pos[1]),
        cfg.r_moon_impact,
        facecolor="gray",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(moon_disk)

    leo_start_orbit = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.r0_earth,
        fill=False,
        linestyle=":",
        linewidth=1.2,
        edgecolor="lightblue",
        alpha=0.8,
    )
    ax.add_patch(leo_start_orbit)

    earth_corridor_min = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.rp_min,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
    )
    ax.add_patch(earth_corridor_min)

    earth_corridor_max = plt.Circle(
        (rE_pos[0], rE_pos[1]),
        cfg.rp_max,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        edgecolor="black",
        alpha=0.95,
    )
    ax.add_patch(earth_corridor_max)

    moon_flyby_upper = plt.Circle(
        (rM_pos[0], rM_pos[1]),
        cfg.r_moon_flyby,
        fill=False,
        linestyle=":",
        linewidth=1.4,
    )
    ax.add_patch(moon_flyby_upper)

    cmap = plt.cm.get_cmap("tab10", max(1, len(sweep_results)))

    text_lines = []

    for i, row in enumerate(sweep_results):
        color = cmap(i)

        traj = np.asarray(row.get("traj", []), dtype=np.float64)
        if traj.size == 0:
            continue

        traj_xy = clip_path_to_radius(traj[:, :2], r_max=1.5)
        if len(traj_xy) >= 2:
            ax.plot(
                traj_xy[:, 0],
                traj_xy[:, 1],
                linewidth=1.4,
                color=color,
                alpha=0.95,
            )

        ballistic_ref_traj = row.get("ballistic_ref_traj", None)
        if ballistic_ref_traj is not None and len(ballistic_ref_traj) > 1:
            ballistic_ref_traj = np.asarray(ballistic_ref_traj, dtype=np.float64)
            ballistic_xy = clip_path_to_radius(ballistic_ref_traj[:, :2], r_max=1.5)

            if len(ballistic_xy) >= 2:
                ax.plot(
                    ballistic_xy[:, 0],
                    ballistic_xy[:, 1],
                    linewidth=1.6,
                    color=color,
                    alpha=0.95,
                )

        spawn_pos = row.get("spawn_pos_rot", None)
        if spawn_pos is not None:
            spawn_pos = np.asarray(spawn_pos, dtype=np.float64).reshape(2,)
            ax.scatter(
                spawn_pos[0],
                spawn_pos[1],
                s=55,
                marker="o",
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=8,
            )

        tli_pos = row.get("tli_pos_rot", None)
        if tli_pos is not None:
            tli_pos = np.asarray(tli_pos, dtype=np.float64).reshape(2,)
            ax.scatter(
                tli_pos[0],
                tli_pos[1],
                s=85,
                marker="x",
                color=color,
                linewidths=2.0,
                zorder=9,
            )

        text_lines.append(
            f"{i+1:02d}: "
            f"spawn={row.get('spawn_theta', np.nan): .4f}, "
            f"tli={row.get('tli_theta', np.nan): .4f}, "
            f"dv0={row.get('dv0', np.nan): .4f}, "
            f"tau={row.get('tli_tau', np.nan): .4f}"
        )

    if len(text_lines) > 0:
        fig.text(
            0.02,
            0.02,
            "\n".join(text_lines[:12]),
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.axis("equal")
    ax.set_xlim(-1.05, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (rotating frame, nondim)")
    ax.set_ylabel("y (rotating frame, nondim)")
    ax.set_title(title if title else "Spawn-theta sweep")

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    if out_path is None:
        out_path = "theta_sweep.png"

    plt.savefig(out_path, dpi=150)
    plt.close(fig)



def _safe_array(x):
    if x is None:
        return np.array([], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def plot_eval_trajectories_grid(
    cfg,
    eval_results,
    title="Eval trajectories",
    out_path=None,
    ncols: int = 4,
    r_max: float = 1.5,
):
    """
    Plot one full eval batch in a grid.
    Each panel shows:
    - ballistic reference branch from the initial condition
    - actual flown trajectory
    - Earth and Moon
    """
    mu = cfg.mu
    rE_pos, rM_pos = earth_moon_positions(mu)

    n = len(eval_results)
    if n == 0:
        return

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    for idx, ep in enumerate(eval_results):
        ax = axes[idx // ncols][idx % ncols]

        traj = _safe_array(ep.get("traj", None))
        ballistic = _safe_array(ep.get("ballistic_ref_traj", None))

        if len(ballistic) > 1:
            ballistic = _clip_xy_path_to_radius(ballistic, r_max=r_max)
            if len(ballistic) > 1:
                ax.plot(
                    ballistic[:, 0],
                    ballistic[:, 1],
                    linestyle=":",
                    linewidth=1.3,
                    color="orange",
                    label="Ballistic",
                )

        if len(traj) > 1:
            traj = _clip_xy_path_to_radius(traj, r_max=r_max)
            if len(traj) > 1:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    linewidth=1.2,
                    color="tab:blue",
                    label="Controlled",
                )

        ballistic_terminal = None
        env_snapshot = ep.get("env_snapshot", None)
        if env_snapshot is not None:
            ballistic_terminal = getattr(env_snapshot, "ballistic_terminal_marker_rot", None)
            terminal_marker = getattr(env_snapshot, "terminal_marker_rot", None)
        else:
            terminal_marker = None

        if ballistic_terminal is not None:
            p = np.asarray(ballistic_terminal, dtype=np.float64).reshape(2,)
            ax.scatter(
                p[0], p[1],
                s=35,
                marker="x",
                color="orange",
                linewidths=1.6,
                zorder=8,
            )

        if terminal_marker is not None:
            p = np.asarray(terminal_marker, dtype=np.float64).reshape(2,)
            ax.scatter(
                p[0], p[1],
                s=40,
                marker="x",
                color="red",
                linewidths=1.8,
                zorder=9,
            )

        earth_disk = plt.Circle(
            (rE_pos[0], rE_pos[1]),
            cfg.r_earth_impact,
            facecolor="tab:blue",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(earth_disk)

        moon_disk = plt.Circle(
            (rM_pos[0], rM_pos[1]),
            cfg.r_moon_impact,
            facecolor="gray",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(moon_disk)

        earth_corridor_min = plt.Circle(
            (rE_pos[0], rE_pos[1]),
            cfg.rp_min,
            fill=False,
            linestyle="--",
            linewidth=0.9,
            edgecolor="black",
            alpha=0.7,
        )
        ax.add_patch(earth_corridor_min)

        earth_corridor_max = plt.Circle(
            (rE_pos[0], rE_pos[1]),
            cfg.rp_max,
            fill=False,
            linestyle="--",
            linewidth=0.9,
            edgecolor="black",
            alpha=0.7,
        )
        ax.add_patch(earth_corridor_max)

        moon_flyby_upper = plt.Circle(
            (rM_pos[0], rM_pos[1]),
            cfg.r_moon_flyby,
            fill=False,
            linestyle=":",
            linewidth=0.9,
            edgecolor="black",
            alpha=0.7,
        )
        ax.add_patch(moon_flyby_upper)

        reason = str(ep.get("reason", ""))
        ballistic_success = bool(ep.get("ballistic_success", False))
        trajectory_success = bool(ep.get("trajectory_success", False))

        if ballistic_success and trajectory_success:
            outcome = "preserved"
        elif ballistic_success and (not trajectory_success):
            outcome = "degraded"
        elif (not ballistic_success) and trajectory_success:
            outcome = "rescued"
        else:
            outcome = "unchanged_bad"

        ax.set_title(
            f"Ep {idx+1}\n{outcome} | {reason}",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.tick_params(labelsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles = [
        plt.Line2D([0], [0], linestyle=":", color="orange", label="Ballistic"),
        plt.Line2D([0], [0], linestyle="-", color="tab:blue", label="Controlled"),
        plt.Line2D([0], [0], marker="x", color="orange", linestyle="None", label="Ballistic end"),
        plt.Line2D([0], [0], marker="x", color="red", linestyle="None", label="Episode end"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is None:
        out_path = "eval_trajectories_grid.png"
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_mean_eval_dv_curve(history, title="Mean delta-v per eval", out_path=None):
    if len(history) == 0:
        return

    steps = np.asarray([row.get("step", np.nan) for row in history], dtype=np.float64)
    dv_mean = np.asarray([row.get("dv_mean", np.nan) for row in history], dtype=np.float64)
    dv_std = np.asarray([row.get("dv_std", np.nan) for row in history], dtype=np.float64)

    finite = np.isfinite(steps) & np.isfinite(dv_mean)
    if not np.any(finite):
        return

    steps = steps[finite]
    dv_mean = dv_mean[finite]
    dv_std = dv_std[finite]

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    ax.plot(steps, dv_mean, linewidth=1.8, label="mean dv_used")

    if np.any(np.isfinite(dv_std)):
        lo = dv_mean - dv_std
        hi = dv_mean + dv_std
        ax.fill_between(steps, lo, hi, alpha=0.18, label="±1 std")

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean eval Δv [nondim]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    if out_path is None:
        out_path = "mean_eval_dv_curve.png"

    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_training_eval_reward_curve(history, title="Mean reward per eval", out_path=None):
    if len(history) == 0:
        return

    steps = np.asarray([row["step"] for row in history], dtype=np.float64)
    rewards = np.asarray([row["reward_mean"] for row in history], dtype=np.float64)

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.plot(steps, rewards, linewidth=1.8)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean eval reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path is None:
        out_path = "training_eval_reward_curve.png"
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_free_return_rates_curve(history, title="Free-return transition rates", out_path=None):
    if len(history) == 0:
        return

    steps = np.asarray([row["step"] for row in history], dtype=np.float64)

    preserved = np.asarray([row["preservation_rate"] for row in history], dtype=np.float64)
    degraded = np.asarray([row["degradation_rate"] for row in history], dtype=np.float64)
    rescued = np.asarray([row["rescue_rate"] for row in history], dtype=np.float64)
    unchanged_bad = np.asarray([row["unchanged_bad_rate"] for row in history], dtype=np.float64)

    fig = plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    ax.plot(steps, preserved, linewidth=1.7, label="preservation_rate")
    ax.plot(steps, degraded, linewidth=1.7, label="degradation_rate")
    ax.plot(steps, rescued, linewidth=1.7, label="rescue_rate")
    ax.plot(steps, unchanged_bad, linewidth=1.7, label="unchanged_bad_rate")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    if out_path is None:
        out_path = "free_return_rates_curve.png"
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_ppo_metrics_curve(history, title="PPO training metrics", out_path=None):
    if len(history) == 0:
        return

    steps = np.asarray([row["step"] for row in history], dtype=np.float64)

    def arr(key):
        vals = []
        for row in history:
            v = row.get(key, np.nan)
            vals.append(np.nan if v is None else float(v))
        return np.asarray(vals, dtype=np.float64)

    metric_groups = [
        ("Policy / clipping", ["approx_kl", "clip_fraction", "clip_range", "policy_gradient_loss"]),
        ("Value / fit", ["loss", "value_loss"]),
        ("Explained variance", ["explained_variance"]),
        ("Entropy", ["entropy_loss"]),
        ("Policy std", ["std"]),
        ("Learning rate", ["learning_rate"]),
    ]

    fig, axes = plt.subplots(len(metric_groups), 1, figsize=(10, 16), sharex=True)

    for ax, (subtitle, keys) in zip(axes, metric_groups):
        used_any = False
        for key in keys:
            y = arr(key)
            if np.all(~np.isfinite(y)):
                continue
            ax.plot(steps, y, linewidth=1.5, label=key)
            used_any = True

        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)
        if used_any:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Training steps")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if out_path is None:
        out_path = "ppo_metrics_curve.png"
    plt.savefig(out_path, dpi=170)
    plt.close(fig)




def save_spawn_theta_sweep_txt(out_dir: Path, stem: str, sweep_results) -> Path:
    out_path = out_dir / f"{stem}_theta_sweep.txt"

    lines = []
    lines.append("=" * 110)
    lines.append("THETA SWEEP SUMMARY")
    lines.append("=" * 110)
    lines.append(
        "idx | spawn_theta | tli_theta | dv0 | tau_raw | ax | ay | reason | ballistic_hit | success"
    )
    lines.append("-" * 110)

    for i, row in enumerate(sweep_results, start=1):
        lines.append(
            f"{i:02d} | "
            f"{row.get('spawn_theta', np.nan): .6f} | "
            f"{row.get('tli_theta', np.nan): .6f} | "
            f"{row.get('dv0', np.nan): .6f} | "
            f"{row.get('tli_tau', np.nan): .6f} | "
            f"{row.get('tli_ax', np.nan): .6f} | "
            f"{row.get('tli_ay', np.nan): .6f} | "
            f"{row.get('reason', '')} | "
            f"{bool(row.get('ballistic_tli_corridor_hit', False))} | "
            f"{bool(row.get('success_flag_latched', False))}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def audit_reward_records(rewards, reward_records, tol: float = 1e-9) -> Dict[str, Any]:
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)

    audit = {
        "passed": True,
        "n_steps": int(len(rewards)),
        "episode_total_from_rewards": float(np.sum(rewards)) if len(rewards) > 0 else 0.0,
        "episode_total_from_step_reward": 0.0,
        "episode_total_from_r_total": 0.0,
        "episode_scalar_minus_step_reward": np.nan,
        "episode_scalar_minus_r_total": np.nan,
        "max_step_reward_minus_r_total_abs": 0.0,
        "missing_reward_record_indices": [],
        "missing_r_total_indices": [],
        "warnings": [],
    }

    if len(reward_records) != len(rewards):
        audit["passed"] = False
        audit["warnings"].append(
            f"reward_records length {len(reward_records)} != rewards length {len(rewards)}"
        )

    step_reward_sum = 0.0
    r_total_sum = 0.0
    max_step_diff = 0.0

    n = min(len(reward_records), len(rewards))
    for i in range(n):
        rec = reward_records[i]
        if not isinstance(rec, dict):
            audit["passed"] = False
            audit["missing_reward_record_indices"].append(i)
            continue

        step_reward = rec.get("step_reward", None)
        terms = rec.get("terms", None)

        if step_reward is None or not np.isfinite(float(step_reward)):
            audit["passed"] = False
            audit["warnings"].append(f"step {i}: missing/invalid step_reward")
            continue

        step_reward = float(step_reward)
        step_reward_sum += step_reward

        if not isinstance(terms, dict):
            audit["passed"] = False
            audit["warnings"].append(f"step {i}: missing terms dict")
            continue

        if "r_total" not in terms:
            audit["passed"] = False
            audit["missing_r_total_indices"].append(i)
            continue

        r_total = float(terms["r_total"])
        r_total_sum += r_total

        step_diff = abs(step_reward - r_total)
        if step_diff > max_step_diff:
            max_step_diff = step_diff

        if step_diff > tol:
            audit["passed"] = False
            audit["warnings"].append(
                f"step {i}: step_reward ({step_reward:.12f}) != r_total ({r_total:.12f})"
            )

    audit["episode_total_from_step_reward"] = float(step_reward_sum)
    audit["episode_total_from_r_total"] = float(r_total_sum)
    audit["episode_scalar_minus_step_reward"] = float(audit["episode_total_from_rewards"] - step_reward_sum)
    audit["episode_scalar_minus_r_total"] = float(audit["episode_total_from_rewards"] - r_total_sum)
    audit["max_step_reward_minus_r_total_abs"] = float(max_step_diff)

    if abs(audit["episode_scalar_minus_step_reward"]) > tol:
        audit["passed"] = False
        audit["warnings"].append(
            "episode total from rewards does not match sum of reward_record.step_reward"
        )

    if abs(audit["episode_scalar_minus_r_total"]) > tol:
        audit["passed"] = False
        audit["warnings"].append(
            "episode total from rewards does not match sum of reward_record.terms['r_total']"
        )

    return audit


def save_episode_report_json(
    out_dir: Path,
    stem: str,
    env,
    rewards,
    terms_ts,
    info_last,
    reward_records=None,
    audit=None,
) -> Path:
    out_path = out_dir / f"{stem}_episode_report.json"

    payload = {
        "episode_summary": {
            "term_reason": info_last.get("term_reason", ""),
            "success": bool(info_last.get("success", False)),
            "flyby_done": bool(info_last.get("flyby_done", False)),
            "return_done": bool(info_last.get("return_done", False)),
            "left_leo": bool(info_last.get("left_leo", False)),
            "total_reward": float(np.sum(np.asarray(rewards, dtype=np.float64))) if len(rewards) > 0 else 0.0,
            "n_steps": int(len(rewards)),
        },
        "info_last": _jsonify(info_last),
        "rewards": _jsonify(np.asarray(rewards, dtype=np.float64)),
        "terms_ts": _jsonify(terms_ts),
        "reward_records": _jsonify(reward_records if reward_records is not None else []),
        "reward_audit": _jsonify(audit if audit is not None else {}),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


def collect_episode_reward_timeseries(env, model, deterministic=True, max_steps=100000):
    obs, info = env.reset()
    done = False
    trunc = False

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    rewards = []
    reward_records = []
    steps = 0

    while not (done or trunc):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        obs, r, done, trunc, info = env.step(action)
        episode_start = np.array([done or trunc], dtype=bool)

        rewards.append(float(r))

        record = info.get("reward_record", None)
        if not isinstance(record, dict):
            raise RuntimeError(
                "Missing info['reward_record'] during rollout. "
                "Reward reporting is not trustworthy."
            )
        reward_records.append(record)

        steps += 1
        if steps >= max_steps:
            break

    all_term_keys = sorted({
        k
        for rec in reward_records
        for k in rec.get("terms", {}).keys()
    })

    T = len(rewards)
    rewards = np.asarray(rewards, dtype=np.float64)

    terms_ts = {}
    for k in all_term_keys:
        arr = np.zeros((T,), dtype=np.float64)
        for i, rec in enumerate(reward_records):
            arr[i] = float(rec.get("terms", {}).get(k, 0.0))
        terms_ts[k] = arr

    all_flag_keys = sorted({
        k
        for rec in reward_records
        for k in rec.get("flags", {}).keys()
    })
    for k in all_flag_keys:
        arr = np.zeros((T,), dtype=np.float64)
        for i, rec in enumerate(reward_records):
            arr[i] = 1.0 if bool(rec.get("flags", {}).get(k, False)) else 0.0
        terms_ts[f"flag_{k}"] = arr

    all_metric_keys = sorted({
        k
        for rec in reward_records
        for k in rec.get("metrics", {}).keys()
        if k != "term_reason"
    })
    for k in all_metric_keys:
        arr = np.zeros((T,), dtype=np.float64)
        arr[:] = np.nan
        for i, rec in enumerate(reward_records):
            val = rec.get("metrics", {}).get(k, np.nan)
            try:
                fv = float(val)
            except Exception:
                fv = np.nan
            arr[i] = fv if np.isfinite(fv) else np.nan
        terms_ts[f"metric_{k}"] = arr

    audit = audit_reward_records(rewards, reward_records)

    if hasattr(env, "_build_ballistic_reference_from_tli"):
        env._build_ballistic_reference_from_tli()

    return rewards, terms_ts, info, reward_records, audit


def build_episode_report_text(env, rewards, terms_ts, info_last, reward_records=None, audit=None) -> str:
    lines = []

    def safe_sum(name: str) -> float:
        if name not in terms_ts:
            return 0.0
        arr = np.asarray(terms_ts[name], dtype=np.float64)
        if arr.size == 0:
            return 0.0
        return float(np.nansum(arr))

    def safe_last(name: str, default=np.nan) -> float:
        if name not in terms_ts:
            return float(default)
        arr = np.asarray(terms_ts[name], dtype=np.float64)
        if arr.size == 0:
            return float(default)
        val = arr[-1]
        return float(val) if np.isfinite(val) else float(default)

    def getf(key: str, default=np.nan) -> float:
        try:
            val = info_last.get(key, default)
            return float(val)
        except Exception:
            return float(default)

    def getb(key: str, default=False) -> bool:
        try:
            return bool(info_last.get(key, default))
        except Exception:
            return bool(default)

    def add_obs_vector(lines_out, obs_vec, indent: str = ""):
        obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
        if obs_vec.size == 0:
            lines_out.append(f"{indent}No observation vector available.")
            return

        obs_labels = [
            "obs[0]  x_scaled",
            "obs[1]  y_scaled",
            "obs[2]  vx_scaled",
            "obs[3]  vy_scaled",
            "obs[4]  rE_scaled",
            "obs[5]  rM_scaled",
            "obs[6]  C_scaled",
            "obs[7]  t_fraction",
            "obs[8]  dv_fraction",
        ]

        cfg_local = getattr(env, "cfg", None)
        add_phase = bool(getattr(cfg_local, "add_phase_angle_obs", False))
        add_legacy = bool(
            getattr(cfg_local, "add_mode_obs", False)
            and getattr(cfg_local, "add_legacy_mode_obs", False)
        )

        if add_phase:
            obs_labels.append(f"obs[{len(obs_labels)}]  phase_angle_norm")

        if add_legacy:
            obs_labels.extend([
                f"obs[{len(obs_labels)+0}]  tli_used_flag",
                f"obs[{len(obs_labels)+1}]  tau_max_current_norm",
                f"obs[{len(obs_labels)+2}]  dv_cap_current_norm",
                f"obs[{len(obs_labels)+3}]  pre_tli_clock_norm",
            ])
        
        add_staged_tli = bool(
            getattr(cfg_local, "add_mode_obs", False)
            and getattr(cfg_local, "add_staged_tli_obs", False)
            and getattr(cfg_local, "staged_tli_enabled", False)
        )

        if add_staged_tli:
            obs_labels.extend([
                f"obs[{len(obs_labels)+0}]  pre_tli_cum_dv_norm",
                f"obs[{len(obs_labels)+1}]  pre_tli_burn_count_norm",
            ])

        n = min(obs_vec.size, len(obs_labels))
        for i in range(n):
            lines_out.append(f"{indent}{obs_labels[i]:<30} = {obs_vec[i]: .6f}")

        if obs_vec.size > len(obs_labels):
            for i in range(len(obs_labels), obs_vec.size):
                lines_out.append(f"{indent}obs[{i}]".ljust(30) + f" = {obs_vec[i]: .6f}")

    total_reward = float(np.sum(np.asarray(rewards, dtype=np.float64))) if len(rewards) > 0 else 0.0

    final_obs = None
    try:
        final_obs = env._get_obs()
    except Exception:
        final_obs = None

    # ------------------------------------------------------
    # Final config snapshot from reward_records
    # This is the correct source for cfg_* and w_* values
    # ------------------------------------------------------
    final_config = {}
    if reward_records is not None and len(reward_records) > 0:
        try:
            final_config = reward_records[-1].get("config", {})
            if not isinstance(final_config, dict):
                final_config = {}
        except Exception:
            final_config = {}

    def cfgf(key: str, default=np.nan) -> float:
        try:
            val = final_config.get(key, default)
            return float(val)
        except Exception:
            return float(default)

    def cfgb(key: str, default=False) -> bool:
        try:
            return bool(final_config.get(key, default))
        except Exception:
            return bool(default)

    tli_used = bool(getattr(env, "tli_used", False))
    tli_executed = bool(getattr(env, "tli_executed", False))
    tli_ballistic_reward_given = bool(getattr(env, "tli_ballistic_reward_given", False))
    ballistic_reward_last = float(getattr(env, "ballistic_tli_reward_last", 0.0))
    ballistic_ref_traj = getattr(env, "ballistic_ref_traj", None)
    ballistic_ref_exists = ballistic_ref_traj is not None and len(ballistic_ref_traj) > 1

    ballistic_min_rM = float(
        info_last.get(
            "ballistic_min_rM",
            info_last.get("ballistic_tli_min_rM", np.nan),
        )
    )

    ballistic_rM_terminal = float(
        info_last.get("ballistic_rM_terminal", np.nan)
    )

    ballistic_rE_terminal = float(
        info_last.get("ballistic_rE_terminal", np.nan)
    )

    lines.append("=" * 90)
    lines.append("EPISODE REPORT")
    lines.append("=" * 90)
    lines.append(f"term_reason                 : {info_last.get('term_reason', '')}")
    lines.append(f"success                     : {getb('success', False)}")
    lines.append(f"flyby_done                  : {getb('flyby_done', False)}")
    lines.append(f"return_done                 : {getb('return_done', False)}")
    lines.append(f"left_leo                    : {getb('left_leo', False)}")
    lines.append("")

    lines.append("REWARD AUDIT")
    lines.append("-" * 90)
    if audit is None:
        lines.append("reward_audit_available      : False")
    else:
        lines.append("reward_audit_available      : True")
        lines.append(f"reward_audit_passed         : {bool(audit.get('passed', False))}")
        lines.append(f"reward_audit_n_steps        : {int(audit.get('n_steps', 0))}")
        lines.append(f"reward_total_from_rewards   : {float(audit.get('episode_total_from_rewards', np.nan)):.12f}")
        lines.append(f"reward_total_from_step_rec  : {float(audit.get('episode_total_from_step_reward', np.nan)):.12f}")
        lines.append(f"reward_total_from_r_total   : {float(audit.get('episode_total_from_r_total', np.nan)):.12f}")
        lines.append(f"scalar_minus_step_rec       : {float(audit.get('episode_scalar_minus_step_reward', np.nan)):.12e}")
        lines.append(f"scalar_minus_r_total        : {float(audit.get('episode_scalar_minus_r_total', np.nan)):.12e}")
        lines.append(f"max_step_minus_r_total_abs  : {float(audit.get('max_step_reward_minus_r_total_abs', np.nan)):.12e}")
        warnings = audit.get("warnings", [])
        lines.append(f"reward_audit_warning_count  : {len(warnings)}")
        for i, w in enumerate(warnings[:10], start=1):
            lines.append(f"reward_audit_warning_{i:02d}   : {w}")
    lines.append("")

    lines.append("TLI / BALLISTIC STATUS")
    lines.append("-" * 90)
    lines.append(f"tli_used                    : {tli_used}")
    lines.append(f"tli_executed                : {tli_executed}")
    lines.append(f"tli_ballistic_reward_given  : {tli_ballistic_reward_given}")
    lines.append(f"ballistic_reward_nonzero    : {bool(abs(ballistic_reward_last) > 0.0)}")
    lines.append(f"ballistic_ref_traj_exists   : {ballistic_ref_exists}")
    lines.append(f"ballistic_min_rM            : {ballistic_min_rM:.10f}")
    lines.append(f"ballistic_rM_terminal       : {ballistic_rM_terminal:.10f}")
    lines.append(f"ballistic_rE_terminal       : {ballistic_rE_terminal:.10f}")
    lines.append(f"tli_ax                      : {float(getattr(env, 'tli_ax', np.nan)):.10f}")
    lines.append(f"tli_ay                      : {float(getattr(env, 'tli_ay', np.nan)):.10f}")
    lines.append(f"tli_step_executed           : {getattr(env, 'tli_step_executed', None)}")
    lines.append("")

    lines.append("MISSION METRICS")
    lines.append("-" * 90)
    lines.append(f"total_reward                : {total_reward:.10f}")
    lines.append(f"steps                       : {len(rewards)}")
    lines.append(f"final_time                  : {getf('t', np.nan):.10f}")
    lines.append(f"dv_used                     : {getf('dv_used', np.nan):.10f}")
    lines.append(f"dv0                         : {getf('dv0', np.nan):.10f}")
    lines.append(f"dv_mcc_total                : {getf('dv_mcc_total', np.nan):.10f}")
    lines.append(f"closest_moon_approach       : {getf('min_rM', np.nan):.10f}")
    lines.append(f"closest_earth_approach      : {getf('min_rE', np.nan):.10f}")
    lines.append(f"closest_postflyby_earth     : {getf('min_rE_postflyby', np.nan):.10f}")
    lines.append(f"best_return_corridor_dist   : {getf('best_postflyby_corridor_dist', np.nan):.10f}")
    lines.append("")

    lines.append("REWARDS")
    lines.append("-" * 90)
    lines.append("EPISODE-LEVEL TOTALS")
    lines.append(f"episode_total_reward             : {total_reward:.10f}")
    lines.append(f"average_step_reward              : {float(np.mean(rewards)) if len(rewards) > 0 else 0.0:.10f}")
    lines.append(f"min_step_reward                  : {float(np.min(rewards)) if len(rewards) > 0 else 0.0:.10f}")
    lines.append(f"max_step_reward                  : {float(np.max(rewards)) if len(rewards) > 0 else 0.0:.10f}")
    lines.append(f"nonzero_reward_steps             : {int(np.sum(np.abs(np.asarray(rewards, dtype=np.float64)) > 0.0))}")
    lines.append("")

    lines.append("MAIN ENV REWARD TOTALS")
    lines.append(f"r_dv_total                       : {safe_sum('r_dv'):.10f}")
    lines.append(f"r_budget_total                   : {safe_sum('r_budget'):.10f}")
    lines.append(f"r_escape_total                   : {safe_sum('r_escape'):.10f}")
    lines.append(f"r_crash_total                    : {safe_sum('r_crash'):.10f}")
    lines.append(f"r_invalid_preflyby_earth_return_total : {safe_sum('r_invalid_preflyby_earth_return'):.10f}")
    lines.append(f"r_flyby_total                    : {safe_sum('r_flyby'):.10f}")
    lines.append(f"r_velocity_total                 : {safe_sum('r_velocity'):.10f}")
    lines.append(f"r_return_total                   : {safe_sum('r_return'):.10f}")
    lines.append(f"r_ballistic_tli_total            : {safe_sum('r_ballistic_tli'):.10f}")
    lines.append(f"r_total_sum                      : {safe_sum('r_total'):.10f}")
    lines.append("")

    lines.append("BALLISTIC TLI REWARD BREAKDOWN")
    lines.append(f"r_tli_ballistic_total            : {safe_sum('r_tli_ballistic_total'):.10f}")
    lines.append(f"r_tli_ballistic_scale            : {safe_sum('r_tli_ballistic_scale'):.10f}")
    lines.append(f"r_tli_ballistic_dv               : {safe_sum('r_tli_ballistic_dv'):.10f}")
    lines.append(f"r_tli_ballistic_budget           : {safe_sum('r_tli_ballistic_budget'):.10f}")
    lines.append(f"r_tli_ballistic_escape           : {safe_sum('r_tli_ballistic_escape'):.10f}")
    lines.append(f"r_tli_ballistic_crash            : {safe_sum('r_tli_ballistic_crash'):.10f}")
    lines.append(f"r_tli_ballistic_flyby            : {safe_sum('r_tli_ballistic_flyby'):.10f}")
    lines.append(f"r_tli_ballistic_velocity         : {safe_sum('r_tli_ballistic_velocity'):.10f}")
    lines.append(f"r_tli_ballistic_return           : {safe_sum('r_tli_ballistic_return'):.10f}")
    lines.append(f"r_tli_ballistic_invalid_preflyby_earth_return : {safe_sum('r_tli_ballistic_invalid_preflyby_earth_return'):.10f}")
    lines.append("")

    lines.append("BALLISTIC TLI TERMINAL FLAGS")
    lines.append(f"ballistic_terminal_suppressed    : {safe_sum('r_tli_ballistic_terminal_rewards_suppressed'):.10f}")
    lines.append(f"ballistic_terminal_escape_flag   : {safe_sum('r_tli_ballistic_terminal_escape_flag'):.10f}")
    lines.append(f"ballistic_terminal_timeout_flag  : {safe_sum('r_tli_ballistic_terminal_timeout_flag'):.10f}")
    lines.append(f"ballistic_terminal_success_flag  : {safe_sum('r_tli_ballistic_terminal_success_flag'):.10f}")
    lines.append("")

    lines.append("TERMINAL REWARD FLAGS")
    lines.append(f"terminal_left_leo_sum            : {safe_sum('terminal_left_leo'):.10f}")
    lines.append(f"terminal_bootstrap_timeout       : {safe_sum('terminal_bootstrap_pre_tli_timeout'):.10f}")
    lines.append(f"terminal_meaningful_moon         : {safe_sum('terminal_meaningful_moon_approach'):.10f}")
    lines.append(f"terminal_flyby_allowed           : {safe_sum('terminal_flyby_reward_allowed'):.10f}")
    lines.append(f"terminal_return_eligible         : {safe_sum('terminal_return_eligible'):.10f}")
    lines.append(f"terminal_corridor_hit            : {safe_sum('terminal_corridor_hit_postflyby'):.10f}")
    lines.append("")

    lines.append("DEBUG REWARD SNAPSHOT FROM FINAL STEP")
    lines.append("-" * 90)
    lines.append(f"metric_reward_model_min_rM       : {safe_last('metric_reward_model_min_rM', np.nan):.10f}")
    lines.append(f"metric_reward_model_v_at_min_rM  : {safe_last('metric_reward_model_v_at_min_rM', np.nan):.10f}")
    lines.append(f"metric_min_rM_env                : {safe_last('metric_min_rM_env', np.nan):.10f}")
    lines.append(f"metric_min_rE_postflyby_env      : {safe_last('metric_min_rE_postflyby_env', np.nan):.10f}")
    lines.append(f"metric_dv_mag_step               : {safe_last('metric_dv_mag_step', np.nan):.10f}")
    lines.append("")

    lines.append("ACTIVE CONFIG AT EPISODE END")
    lines.append("-" * 90)
    lines.append(f"cfg_tli_only_mode                : {cfgb('cfg_tli_only_mode', False)}")
    lines.append(f"cfg_reward_after_tli             : {cfgb('cfg_reward_after_tli_ballistic_enabled', False)}")
    lines.append(f"cfg_dv_noise_sigma_tli           : {cfgf('cfg_dv_noise_sigma_tli', np.nan):.10f}")
    lines.append(f"cfg_dv_noise_sigma_mcc           : {cfgf('cfg_dv_noise_sigma_mcc', np.nan):.10f}")
    lines.append("")
    lines.append(f"cfg_r_moon_flyby                 : {cfgf('cfg_r_moon_flyby', np.nan):.10f}")
    lines.append(f"cfg_rp_min                       : {cfgf('cfg_rp_min', np.nan):.10f}")
    lines.append(f"cfg_rp_max                       : {cfgf('cfg_rp_max', np.nan):.10f}")
    lines.append("")
    lines.append(f"w_flyby_final_step               : {cfgf('w_flyby', np.nan):.10f}")
    lines.append(f"w_velocity_final_step            : {cfgf('w_velocity', np.nan):.10f}")
    lines.append(f"w_dv_final_step                  : {cfgf('w_dv', np.nan):.10f}")
    lines.append(f"w_return_final_step              : {cfgf('w_return', np.nan):.10f}")
    lines.append(f"w_budget_final_step              : {cfgf('w_budget', np.nan):.10f}")
    lines.append(f"w_escape_final_step              : {cfgf('w_escape', np.nan):.10f}")
    lines.append(f"w_earth_crash_final_step         : {cfgf('w_earth_crash', np.nan):.10f}")
    lines.append(f"w_moon_crash_final_step          : {cfgf('w_moon_crash', np.nan):.10f}")
    lines.append(f"w_postflyby_earth_crash_final    : {cfgf('w_postflyby_earth_crash', np.nan):.10f}")
    lines.append(f"w_invalid_preflyby_earth_return  : {cfgf('w_invalid_preflyby_earth_return', np.nan):.10f}")
    lines.append("")

    if final_obs is not None:
        lines.append("FINAL OBSERVATION")
        lines.append("-" * 90)
        add_obs_vector(lines, final_obs, indent="")

    lines.append("")
    lines.append("ACTION HISTORY")
    lines.append("-" * 90)
    if len(getattr(env, "action_history", [])) == 0:
        lines.append("No actions logged.")
    else:
        for row in env.action_history:
            lines.append(
                f"step={row.get('step_idx', -1):>4} | "
                f"time={row.get('time', np.nan):.6f} | "
                f"ax={row.get('ax_raw', np.nan): .6f} | "
                f"ay={row.get('ay_raw', np.nan): .6f} | "
                f"tau_raw={row.get('tau_raw', np.nan): .6f} | "
                f"burn={row.get('burn_kind', 'NONE')} | "
                f"dv={row.get('dv_mag', np.nan):.6f} | "
                f"dt={row.get('dt_effective', np.nan):.6f} | "
                f"reward={row.get('reward', np.nan): .6f}"
            )

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


def save_episode_report_txt(
    out_dir: Path,
    stem: str,
    env,
    rewards,
    terms_ts,
    info_last,
    reward_records=None,
    audit=None,
) -> Path:
    txt = build_episode_report_text(
        env=env,
        rewards=rewards,
        terms_ts=terms_ts,
        info_last=info_last,
        reward_records=reward_records,
        audit=audit,
    )
    out_path = out_dir / f"{stem}_episode_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)

    save_episode_report_json(
        out_dir=out_dir,
        stem=stem,
        env=env,
        rewards=rewards,
        terms_ts=terms_ts,
        info_last=info_last,
        reward_records=reward_records,
        audit=audit,
    )

    return out_path