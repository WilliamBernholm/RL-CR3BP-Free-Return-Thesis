"""
validate_ppo_vs_nominal_trajectories_FIXED.py

Fixed validation rerun script.

Key fixes compared with the first validation script:
1) Loads the same custom recurrent PPO class used by sensitivity_analysis_v2.py.
2) Reconstructs the PPO environment config from the saved policy.
3) Repairs the PPO observation layout to match the checkpoint:
   - PPO-TLI normally expects 12 observations because staged-TLI obs are enabled.
   - PPO-MCC normally expects 10 observations.
4) Uses separate environments for:
   - PPO rerun with the PPO-trained config
   - nominal replay with the one-impulse nominal-reference config

Outputs:
    sensitivity analysis/ppo_vs_nominal_validation_fixed_<timestamp>/
        tli_ppo_trajectory_overlay.png/.pdf
        tli_nominal_trajectory_overlay.png/.pdf
        mcc_ppo_trajectory_overlay.png/.pdf
        mcc_nominal_trajectory_overlay.png/.pdf
        tli_validation_samples.csv
        mcc_validation_samples.csv
        validation_summary.txt
        validation_metadata.json
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RUN, CR3BPConfig, RewardConfig, RewardWeights
from cr3bp_env_v4 import (
    CR3BPFreeReturnEnv,
    SeanStyleReward,
    cr3bp_vstar_kms,
    kms_to_nondim_dv,
    earth_moon_positions,
    get_obs_schema,
)

from train_ppo_v4 import build_cfg_and_weights_from_policy


# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_TAG = "ppo_vs_nominal_validation_fixed"

TLI_SENSITIVITY_FOLDER = r"sensitivity analysis\PPOA_2026-05-22_08-51-37_run__Model__stage03_step00757760_R139.38_SR1.000_LD1.00431_CMnan__2026-05-23_17-09-05__mode_tli__N500__2026-06-02_16-52-34"
MCC_SENSITIVITY_FOLDER = r"sensitivity analysis\PPOB_2026-05-08_10-56-47_run__PPOB__stage_3_done_step_602112__2026-05-10_01-42-13__mode_mcc__N500__2026-06-02_10-37-13"

TLI_REFERENCE_JSON = r"sensitivity analysis\strict_nominal_optimizer_2026-06-05_10-58-12\best_tli_solution.json"
MCC_REFERENCE_JSON = r"sensitivity analysis\strict_nominal_optimizer_2026-06-05_10-58-12\best_mcc_solution.json"

# Set manually if needed. Auto-discovery currently found the correct MCC and a nearby TLI checkpoint.
# To force exact TLI, paste exact path here.
PPO_TLI_MODEL_PATH: Optional[str] = r"C:\Users\willi\MEX\PPO LSTM CR3BP\PPO LSTM CR3BP V4\Saved Policies\PPOA_2026-05-22_08-51-37_run\Model__stage03_step00757760_R139.38_SR1.000_LD1.00431_CMnan__2026-05-23_17-09-05"
PPO_MCC_MODEL_PATH: Optional[str] = r"C:\Users\willi\MEX\PPO LSTM CR3BP\PPO LSTM CR3BP V4\Saved Policies\PPOB_2026-05-08_10-56-47_run\PPOB__stage_3_done_step_602112__2026-05-10_01-42-13.zip"

AUTO_TLI_MODEL_KEYWORDS = ["PPOA", "stage03", "step00757760"]
AUTO_MCC_MODEL_KEYWORDS = ["PPOB", "stage_3_done_step_602112"]

MAX_SAMPLES_PER_MODE: Optional[int] = None
MAX_TRAJ_PER_PLOT: Optional[int] = None


MAX_TRAJ_PER_PLOT = 160
MAX_POINTS_PER_TRAJECTORY = 1200
BALANCE_SUCCESS_FAILURE_IN_PLOT = True

MAX_STEPS_TLI = 100000
MAX_STEPS_MCC = 100000

ZERO_BURN_TAU_RAW = 1.0
SEED = 999
PRINT_EVERY = 50

TRAJ_ALPHA = 0.12
TRAJ_LINEWIDTH = 0.55

PLOT_FLYBY_RADIUS_ND = 0.10
PLOT_RETURN_OUTER_ND = 0.10
PLOT_RETURN_INNER_ND = None


# =============================================================================
# BASIC HELPERS
# =============================================================================

def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(script_dir: Path, p: str | Path) -> Path:
    q = Path(p)
    if not q.is_absolute():
        q = script_dir / q
    return q.resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, Path):
            return str(x)
        return str(x)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=conv)


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes", "y")


def parse_float(row: Dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def m_to_nd_pos(meters: float) -> float:
    return (float(meters) / 1000.0) / float(RUN.cr3bp_Lstar_km)


def mps_to_nd_vel(mps: float) -> float:
    return (float(mps) / 1000.0) / float(cr3bp_vstar_kms())


def read_sample_rows(folder: Path) -> List[Dict[str, str]]:
    csv_path = folder / "sample_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find sample_results.csv in:\n{folder}")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if MAX_SAMPLES_PER_MODE is not None:
        rows = rows[:int(MAX_SAMPLES_PER_MODE)]
    return rows


def auto_find_policy(script_dir: Path, keywords: List[str]) -> Path:
    root = script_dir / "Saved Policies"
    if not root.exists():
        raise FileNotFoundError(f"Could not find Saved Policies folder:\n{root}")

    zips = list(root.rglob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip policies found under:\n{root}")

    def score(p: Path) -> int:
        s = str(p).lower()
        return sum(1 for k in keywords if str(k).lower() in s)

    candidates = [(score(p), p) for p in zips]
    candidates = [x for x in candidates if x[0] > 0]
    if not candidates:
        raise FileNotFoundError(
            f"No model matched keywords {keywords}. Set PPO_TLI_MODEL_PATH or PPO_MCC_MODEL_PATH manually."
        )

    candidates.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return candidates[0][1].resolve()


def load_custom_model(policy_path: Path):
    from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2
    return TimeAwareRecurrentPPOv2.load(str(policy_path), device=RUN.device)


# =============================================================================
# PPO CONFIG REPAIR copied from sensitivity_analysis_v2.py logic
# =============================================================================

def force_physical_dv_caps_from_run_config(cfg: CR3BPConfig) -> None:
    if getattr(RUN, "tli_dv_max_kms", None) is not None:
        cfg.dv_max_tli = float(kms_to_nondim_dv(float(RUN.tli_dv_max_kms)))
    if getattr(RUN, "mcc_dv_max_kms", None) is not None:
        cfg.dv_max_mcc = float(kms_to_nondim_dv(float(RUN.mcc_dv_max_kms)))


def repair_cfg_observation_space_to_model(cfg: CR3BPConfig, model, mode: str) -> None:
    expected = int(model.observation_space.shape[0])

    cfg.add_phase_angle_obs = True
    cfg.add_mode_obs = True
    cfg.add_legacy_mode_obs = False

    force_physical_dv_caps_from_run_config(cfg)

    if expected == 12:
        cfg.trainer_mode = "ppo_a"
        cfg.tli_only_mode = True
        cfg.reward_after_tli_ballistic_enabled = True
        cfg.mcc_enabled = True

        cfg.staged_tli_enabled = True
        cfg.add_staged_tli_obs = True
        cfg.staged_tli_commit_on_cumulative_dv = True
        cfg.staged_tli_limit_burn_count = True

        cfg.staged_tli_max_burn_count = 60
        cfg.staged_tli_cumulative_dv_target = float(kms_to_nondim_dv(3.1))
        cfg.staged_tli_min_commit_frac_of_target = 1.0

    elif expected == 10:
        if mode == "mcc":
            if str(getattr(cfg, "trainer_mode", "")).lower() not in (
                "ppo_b_library", "ppo_b_baseline", "ppo_b_from_external_ic"
            ):
                cfg.trainer_mode = "ppo_b_library"
            cfg.tli_only_mode = False
            cfg.mcc_enabled = True

        cfg.staged_tli_enabled = False
        cfg.add_staged_tli_obs = False
        cfg.add_legacy_mode_obs = False

    else:
        raise ValueError(
            f"Loaded model expects observation dimension {expected}, but this script only knows "
            f"the current 10D PPO-MCC and 12D PPO-TLI layouts."
        )


def apply_eval_settings_to_ppo_cfg(cfg: CR3BPConfig, mode: str, tli_ref: Dict[str, Any], mcc_ref: Dict[str, Any]) -> None:
    if mode == "tli":
        theta = float(tli_ref["theta_rad"])
        cfg.spawn_theta_limit_enabled = True
        cfg.spawn_theta_min = theta
        cfg.spawn_theta_max = theta
        cfg.trainer_mode = "ppo_a"
        cfg.tli_only_mode = True
        cfg.reward_after_tli_ballistic_enabled = True
    else:
        cfg.trainer_mode = "ppo_b_library"
        cfg.tli_only_mode = False
        cfg.mcc_enabled = True
        cfg.ppo_b_case_source = "scenario_library"
        cfg.ppo_b_library_path = str(mcc_ref["library_path"])
        cfg.ppo_b_use_fixed_index = True
        cfg.ppo_b_fixed_index = int(mcc_ref["library_index"])
        cfg.ppo_b_prob_good = 0.0
        cfg.ppo_b_prob_savable = 1.0
        cfg.ppo_b_prob_bad = 0.0
        cfg.ppo_b_eval_use_same_distribution = True
        cfg.ppo_b_fixed_state_noise_pos = 0.0
        cfg.ppo_b_fixed_state_noise_vel = 0.0


# =============================================================================
# NOMINAL CONFIGS
# =============================================================================

def make_nominal_tli_cfg(tli_ref: Dict[str, Any]) -> CR3BPConfig:
    cfg = CR3BPConfig()
    cfg.trainer_mode = "ppo_a"
    cfg.tli_control_mode = "full"
    cfg.tli_only_mode = True
    cfg.mcc_enabled = False
    cfg.reward_after_tli_ballistic_enabled = True

    theta = float(tli_ref["theta_rad"])
    cfg.spawn_theta_limit_enabled = True
    cfg.spawn_theta_min = theta
    cfg.spawn_theta_max = theta

    cfg.staged_tli_enabled = False
    cfg.add_staged_tli_obs = False
    cfg.staged_tli_commit_on_cumulative_dv = False

    ref_dv_kms = float(tli_ref.get("dv_kms", float(tli_ref.get("dv_mps", 3100.0)) / 1000.0))
    cfg.dv_max_tli = float(kms_to_nondim_dv(max(3.4, ref_dv_kms)))
    cfg.dv_max_mcc = float(kms_to_nondim_dv(0.05))

    cfg.add_phase_angle_obs = True
    cfg.add_mode_obs = True
    cfg.add_legacy_mode_obs = False
    return cfg


def make_nominal_mcc_cfg(mcc_ref: Dict[str, Any]) -> CR3BPConfig:
    cfg = CR3BPConfig()
    cfg.trainer_mode = "ppo_b_library"
    cfg.tli_control_mode = "full"
    cfg.tli_only_mode = False
    cfg.mcc_enabled = True
    cfg.reward_after_tli_ballistic_enabled = False

    cfg.ppo_b_case_source = "scenario_library"
    cfg.ppo_b_library_path = str(mcc_ref["library_path"])
    cfg.ppo_b_use_fixed_index = True
    cfg.ppo_b_fixed_index = int(mcc_ref["library_index"])
    cfg.ppo_b_prob_good = 0.0
    cfg.ppo_b_prob_savable = 1.0
    cfg.ppo_b_prob_bad = 0.0
    cfg.ppo_b_eval_use_same_distribution = True
    cfg.ppo_b_fixed_state_noise_pos = 0.0
    cfg.ppo_b_fixed_state_noise_vel = 0.0

    ref_dv_mps = float(mcc_ref.get("dv_mps", 23.5))
    cfg.dv_max_mcc = float(kms_to_nondim_dv(max(0.060, ref_dv_mps / 1000.0)))
    cfg.dv_max_tli = float(kms_to_nondim_dv(0.4))

    cfg.add_phase_angle_obs = True
    cfg.add_mode_obs = True
    cfg.add_legacy_mode_obs = False
    cfg.staged_tli_enabled = False
    cfg.add_staged_tli_obs = False
    return cfg


def make_env(cfg: CR3BPConfig, weights: RewardWeights, seed: int) -> CR3BPFreeReturnEnv:
    env = CR3BPFreeReturnEnv(
        cfg,
        seed=seed,
        reward_model=SeanStyleReward(RewardConfig(), weights),
    )
    if hasattr(env, "set_debug_eval"):
        env.set_debug_eval(True)
    return env


# =============================================================================
# RESET / ACTION / CLASSIFICATION
# =============================================================================

def refresh_env_after_state_edit(env: CR3BPFreeReturnEnv) -> None:
    env.traj = [np.asarray(env.state, dtype=np.float64).copy()]
    env.t_hist = [float(getattr(env, "t", 0.0))]
    env.action_history = []
    env.burns = []
    if hasattr(env, "burn_events"):
        env.burn_events = []
    if hasattr(env, "mcc_ballistic_overlays"):
        env.mcc_ballistic_overlays = []
    if hasattr(env, "info_last"):
        env.info_last = {}


def reset_and_perturb(env: CR3BPFreeReturnEnv, mode: str, tli_ref: Dict[str, Any],
                      dx_m: float, dy_m: float, dvx_mps: float, dvy_mps: float) -> None:
    options: Dict[str, Any] = {}
    if mode == "tli":
        options["forced_spawn_theta"] = float(tli_ref["theta_rad"])

    env.reset(options=options if options else None)
    state_nominal = np.asarray(env.state, dtype=np.float64).copy()
    env.state = state_nominal.copy()

    env.state[0] += m_to_nd_pos(dx_m)
    env.state[1] += m_to_nd_pos(dy_m)
    env.state[2] += mps_to_nd_vel(dvx_mps)
    env.state[3] += mps_to_nd_vel(dvy_mps)

    refresh_env_after_state_edit(env)


def action_from_dv(dv_nd: float, angle_rad: float, dv_cap_nd: float, tau_raw: float) -> np.ndarray:
    frac = float(dv_nd) / max(float(dv_cap_nd), 1e-12)
    frac = float(np.clip(frac, 0.0, 1.0))
    return np.array(
        [frac * math.cos(angle_rad), frac * math.sin(angle_rad), float(np.clip(tau_raw, -1.0, 1.0))],
        dtype=np.float64,
    )


def zero_action() -> np.ndarray:
    return np.array([0.0, 0.0, float(np.clip(ZERO_BURN_TAU_RAW, -1.0, 1.0))], dtype=np.float64)


def ref_angle_rad(ref: Dict[str, Any]) -> float:
    if "angle_rad" in ref:
        return float(ref["angle_rad"])
    return math.radians(float(ref["angle_deg"]))


def build_reference_action(mode: str, cfg: CR3BPConfig, ref: Dict[str, Any]) -> np.ndarray:
    if mode == "tli":
        dv_nd = kms_to_nondim_dv(float(ref.get("dv_kms", ref.get("dv_mps", 3100.0) / 1000.0)))
        tau_raw = float(ref.get("tau_raw", -1.0))
        return action_from_dv(dv_nd, ref_angle_rad(ref), float(cfg.dv_max_tli), tau_raw)

    dv_nd = kms_to_nondim_dv(float(ref.get("dv_mps", 23.5)) / 1000.0)
    tau_raw = float(ref.get("tau_raw", 1.0))
    return action_from_dv(dv_nd, ref_angle_rad(ref), float(cfg.dv_max_mcc), tau_raw)


def finite_or_nan(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def audit_trajectory_events(traj: np.ndarray, cfg: CR3BPConfig) -> Dict[str, Any]:
    """
    Independent trajectory audit.

    This scans the stored trajectory itself and checks:
      - lunar flyby radius entry
      - Earth return corridor entry
      - Earth impact
      - Moon impact
      - whether impacts happened after flyby/corridor

    This is intentionally independent from env.success.
    """

    tr = np.asarray(traj, dtype=float)

    out = {
        "audit_has_traj": False,

        "audit_flyby_entered": False,
        "audit_corridor_entered": False,
        "audit_corridor_exited_outward": False,

        "audit_earth_impact_any": False,
        "audit_moon_impact_any": False,

        "audit_earth_impact_after_flyby": False,
        "audit_earth_impact_after_corridor": False,
        "audit_moon_impact_after_flyby": False,
        "audit_success_then_earth_impact": False,

        "audit_min_rE_nd": np.nan,
        "audit_min_rM_nd": np.nan,
        "audit_min_rE_postflyby_nd": np.nan,

        "audit_flyby_idx": -1,
        "audit_corridor_idx": -1,
        "audit_corridor_exit_idx": -1,
        "audit_earth_impact_idx": -1,
        "audit_moon_impact_idx": -1,

        "audit_clean_success": False,
    }

    if tr.ndim != 2 or tr.shape[0] < 2:
        return out

    out["audit_has_traj"] = True

    mu = float(getattr(cfg, "mu", 0.0121505856))
    rE_pos, rM_pos = earth_moon_positions(mu)

    pos = tr[:, :2]

    rE = np.linalg.norm(pos - rE_pos.reshape(1, 2), axis=1)
    rM = np.linalg.norm(pos - rM_pos.reshape(1, 2), axis=1)

    out["audit_min_rE_nd"] = float(np.nanmin(rE))
    out["audit_min_rM_nd"] = float(np.nanmin(rM))

    r_flyby = float(getattr(cfg, "r_moon_flyby", 0.1))
    rp_min = float(getattr(cfg, "rp_min", 0.0))
    rp_max = float(getattr(cfg, "rp_max", 0.1))
    r_earth_impact = float(getattr(cfg, "r_earth_impact", 0.017))
    r_moon_impact = float(getattr(cfg, "r_moon_impact", 0.0045))

    flyby_indices = np.where(rM <= r_flyby)[0]
    if len(flyby_indices) > 0:
        flyby_idx = int(flyby_indices[0])
        out["audit_flyby_entered"] = True
        out["audit_flyby_idx"] = flyby_idx
        out["audit_min_rE_postflyby_nd"] = float(np.nanmin(rE[flyby_idx:]))

        corridor_indices = np.where((np.arange(len(rE)) >= flyby_idx) & (rE >= rp_min) & (rE <= rp_max))[0]
        if len(corridor_indices) > 0:
            corridor_idx = int(corridor_indices[0])
            out["audit_corridor_entered"] = True
            out["audit_corridor_idx"] = corridor_idx

            # Outward exit: after entering corridor, rE goes outside rp_max while increasing.
            for j in range(corridor_idx + 1, len(rE)):
                if rE[j] > rp_max and rE[j] > rE[j - 1]:
                    out["audit_corridor_exited_outward"] = True
                    out["audit_corridor_exit_idx"] = int(j)
                    break

    earth_indices = np.where(rE <= r_earth_impact)[0]
    if len(earth_indices) > 0:
        earth_idx = int(earth_indices[0])
        out["audit_earth_impact_any"] = True
        out["audit_earth_impact_idx"] = earth_idx

        if out["audit_flyby_idx"] >= 0 and earth_idx >= out["audit_flyby_idx"]:
            out["audit_earth_impact_after_flyby"] = True

        if out["audit_corridor_idx"] >= 0 and earth_idx >= out["audit_corridor_idx"]:
            out["audit_earth_impact_after_corridor"] = True

    moon_indices = np.where(rM <= r_moon_impact)[0]
    if len(moon_indices) > 0:
        moon_idx = int(moon_indices[0])
        out["audit_moon_impact_any"] = True
        out["audit_moon_impact_idx"] = moon_idx

        if out["audit_flyby_idx"] >= 0 and moon_idx >= out["audit_flyby_idx"]:
            out["audit_moon_impact_after_flyby"] = True

    out["audit_success_then_earth_impact"] = bool(
        out["audit_flyby_entered"]
        and out["audit_corridor_entered"]
        and out["audit_corridor_exited_outward"]
        and out["audit_earth_impact_after_corridor"]
    )

    out["audit_clean_success"] = bool(
        out["audit_flyby_entered"]
        and out["audit_corridor_entered"]
        and out["audit_corridor_exited_outward"]
        and not out["audit_earth_impact_any"]
        and not out["audit_moon_impact_any"]
    )

    return out


def classify_episode(mode: str, info: Dict[str, Any], reason: str, cfg: CR3BPConfig) -> Dict[str, Any]:
    reason_l = str(reason).lower()

    earth_impact = bool("earth" in reason_l and "impact" in reason_l)
    moon_impact = bool("moon" in reason_l and "impact" in reason_l)
    escape = bool(reason_l == "escape")
    invalid = bool(reason_l == "invalid_preflyby_earth_return")

    if mode == "tli":
        corridor_success = bool(
            info.get("ballistic_tli_corridor_hit", False)
            or info.get("ballistic_corridor_hit", False)
            or info.get("return_corridor_hit_postflyby", False)
        )
        mission_success = bool(
            info.get("ballistic_tli_success", False)
            or info.get("success", False)
            or (corridor_success and not earth_impact and not moon_impact and not escape and not invalid)
        )
        flyby_done = bool(
            np.isfinite(finite_or_nan(info.get("ballistic_tli_min_rM", np.nan)))
            and finite_or_nan(info.get("ballistic_tli_min_rM", np.nan)) <= float(getattr(cfg, "r_moon_flyby", 0.1))
        )
    else:
        flyby_done = bool(info.get("flyby_done", False))
        corridor_success = bool(info.get("return_corridor_hit_postflyby", False))
        mission_success = bool(info.get("success", False) or reason_l == "success")

    broad_success = bool(mission_success or (flyby_done and corridor_success))

    return {
        "success": bool(broad_success),
        "mission_success": bool(mission_success),
        "flyby_done": bool(flyby_done),
        "corridor_success": bool(corridor_success),
        "earth_impact": bool(earth_impact),
        "moon_impact": bool(moon_impact),
        "escape": bool(escape),
        "invalid_preflyby_return": bool(invalid),
    }


def extract_traj(mode: str, env: CR3BPFreeReturnEnv) -> np.ndarray:
    if mode == "tli" and hasattr(env, "ballistic_ref_traj"):
        tr = getattr(env, "ballistic_ref_traj", None)
        if tr is not None and len(tr) > 1:
            return np.asarray(tr, dtype=float)
    if hasattr(env, "traj") and len(env.traj) > 1:
        return np.asarray(env.traj, dtype=float)
    return np.zeros((0, 4), dtype=float)


def run_nominal_episode(env: CR3BPFreeReturnEnv, mode: str, first_action: np.ndarray, max_steps: int) -> Tuple[Dict[str, Any], np.ndarray]:
    rewards: List[float] = []
    last_info: Dict[str, Any] = {}
    done = False
    trunc = False

    for k in range(int(max_steps)):
        action = first_action if k == 0 else zero_action()
        _obs, r, done, trunc, last_info = env.step(action)
        rewards.append(float(r))
        if done or trunc:
            break

    traj = extract_traj(mode, env)

    reason = str(last_info.get("term_reason", "max_steps" if not (done or trunc) else "unknown"))

    cls = classify_episode(mode, last_info, reason, env.cfg)
    audit = audit_trajectory_events(traj, env.cfg)

    cls["reason"] = reason
    cls["reward_sum"] = float(np.sum(rewards)) if rewards else 0.0
    cls["n_steps"] = int(len(rewards))

    cls.update(audit)

    # This is the one you should use for plotting/reporting.
    cls["clean_success_no_impact"] = bool(
        cls.get("success", False)
        and not cls.get("audit_earth_impact_any", False)
        and not cls.get("audit_moon_impact_any", False)
    )

    return cls, traj


def run_ppo_from_current_state(model: Any, env: CR3BPFreeReturnEnv, mode: str, max_steps: int) -> Tuple[Dict[str, Any], np.ndarray]:
    obs = env._get_obs()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    rewards: List[float] = []
    last_info: Dict[str, Any] = {}
    done = False
    trunc = False

    for _ in range(int(max_steps)):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        episode_starts = np.array([False], dtype=bool)
        obs, r, done, trunc, last_info = env.step(action)
        rewards.append(float(r))
        if done or trunc:
            break

    traj = extract_traj(mode, env)

    reason = str(last_info.get("term_reason", "max_steps" if not (done or trunc) else "unknown"))

    cls = classify_episode(mode, last_info, reason, env.cfg)
    audit = audit_trajectory_events(traj, env.cfg)

    cls["reason"] = reason
    cls["reward_sum"] = float(np.sum(rewards)) if rewards else 0.0
    cls["n_steps"] = int(len(rewards))

    cls.update(audit)

    cls["clean_success_no_impact"] = bool(
        cls.get("success", False)
        and not cls.get("audit_earth_impact_any", False)
        and not cls.get("audit_moon_impact_any", False)
    )

    return cls, traj


# =============================================================================
# PLOTTING
# =============================================================================

@dataclass
class TrajectoryRecord:
    mode: str
    controller: str
    sample_idx: int
    sigma_pos_m: float
    sigma_vel_mps: float
    success: bool
    reason: str
    traj: np.ndarray


def downsample_traj(tr: np.ndarray, max_points: int) -> np.ndarray:
    tr = np.asarray(tr, dtype=float)

    if tr.ndim != 2 or tr.shape[0] < 2:
        return np.zeros((0, 4), dtype=float)

    if tr.shape[0] <= int(max_points):
        return tr

    stride = int(np.ceil(tr.shape[0] / int(max_points)))
    return tr[::stride]


def select_records_for_plot(records: List[TrajectoryRecord], max_records: int) -> List[TrajectoryRecord]:
    if len(records) <= int(max_records):
        return records

    rng = np.random.default_rng(SEED)

    if not BALANCE_SUCCESS_FAILURE_IN_PLOT:
        idx = rng.choice(len(records), size=int(max_records), replace=False)
        return [records[i] for i in idx]

    success_records = [r for r in records if r.success]
    failure_records = [r for r in records if not r.success]

    n_success = min(len(success_records), int(max_records) // 2)
    n_failure = min(len(failure_records), int(max_records) - n_success)

    # Fill remaining slots if one class has too few samples.
    remaining = int(max_records) - n_success - n_failure
    if remaining > 0:
        if len(success_records) - n_success > len(failure_records) - n_failure:
            n_success += min(remaining, len(success_records) - n_success)
        else:
            n_failure += min(remaining, len(failure_records) - n_failure)

    chosen = []

    if n_success > 0:
        idx_s = rng.choice(len(success_records), size=n_success, replace=False)
        chosen.extend([success_records[i] for i in idx_s])

    if n_failure > 0:
        idx_f = rng.choice(len(failure_records), size=n_failure, replace=False)
        chosen.extend([failure_records[i] for i in idx_f])

    return chosen


def plot_overlay(out_dir: Path, name: str, cfg: CR3BPConfig, records: List[TrajectoryRecord]) -> None:
    mu = float(getattr(cfg, "mu", 0.0121505856))
    rE_pos, rM_pos = earth_moon_positions(mu)

    r_earth_impact = float(getattr(cfg, "r_earth_impact", 0.017))
    r_moon_impact = float(getattr(cfg, "r_moon_impact", 0.0045))

    return_inner = (
        float(PLOT_RETURN_INNER_ND)
        if PLOT_RETURN_INNER_ND is not None
        else r_earth_impact
    )
    return_outer = float(PLOT_RETURN_OUTER_ND)

    n_total = len(records)
    n_success_total = sum(r.success for r in records)
    sr_total = 100.0 * n_success_total / max(1, n_total)

    plot_records = select_records_for_plot(records, int(MAX_TRAJ_PER_PLOT))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)

    xmin = ymin = np.inf
    xmax = ymax = -np.inf

    # Plot failures first, successes on top.
    for success_flag in [False, True]:
        for rec in plot_records:
            if rec.success != success_flag:
                continue

            tr = downsample_traj(rec.traj, int(MAX_POINTS_PER_TRAJECTORY))
            if tr.shape[0] < 2:
                continue

            xy = tr[:, :2]
            good = np.all(np.isfinite(xy), axis=1)
            xy = xy[good]

            if len(xy) < 2:
                continue

            # Plot only reasonable region for axis calculation.
            r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
            xy_for_lim = xy[r <= 2.5] if np.any(r <= 2.5) else xy

            xmin = min(xmin, float(np.nanmin(xy_for_lim[:, 0])))
            xmax = max(xmax, float(np.nanmax(xy_for_lim[:, 0])))
            ymin = min(ymin, float(np.nanmin(xy_for_lim[:, 1])))
            ymax = max(ymax, float(np.nanmax(xy_for_lim[:, 1])))

            color = "green" if rec.success else "red"
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                alpha=TRAJ_ALPHA,
                lw=TRAJ_LINEWIDTH,
            )

    # Bodies
    ax.scatter([rE_pos[0]], [rE_pos[1]], s=160, color="tab:blue", label="Earth", zorder=5)
    ax.scatter([rM_pos[0]], [rM_pos[1]], s=90, color="tab:orange", label="Moon", zorder=5)

    # Mission geometry
    ax.add_patch(
        plt.Circle(
            (rE_pos[0], rE_pos[1]),
            r_earth_impact,
            fill=False,
            lw=1.0,
            ls=":",
            color="black",
            label="Earth impact",
        )
    )

    ax.add_patch(
        plt.Circle(
            (rM_pos[0], rM_pos[1]),
            r_moon_impact,
            fill=False,
            lw=1.0,
            ls=":",
            color="black",
            label="Moon impact",
        )
    )

    ax.add_patch(
        plt.Circle(
            (rM_pos[0], rM_pos[1]),
            PLOT_FLYBY_RADIUS_ND,
            fill=False,
            lw=1.4,
            color="black",
            label="Flyby radius 0.1",
        )
    )

    ax.add_patch(
        plt.Circle(
            (rE_pos[0], rE_pos[1]),
            return_inner,
            fill=False,
            lw=1.1,
            ls="--",
            color="black",
            label="Return lower",
        )
    )

    ax.add_patch(
        plt.Circle(
            (rE_pos[0], rE_pos[1]),
            return_outer,
            fill=False,
            lw=1.4,
            ls="--",
            color="black",
            label="Return upper 0.1",
        )
    )

    # Legend handles
    ax.plot([], [], color="green", lw=2.0, label="Success")
    ax.plot([], [], color="red", lw=2.0, label="Failure")

    ax.set_title(
        f"{name}\n"
        f"Full set: N={n_total}, success={n_success_total}, SR={sr_total:.1f}% | "
        f"Plotted: {len(plot_records)} trajectories"
    )

    ax.set_xlabel("x [nondim]")
    ax.set_ylabel("y [nondim]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")

    if np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax):
        pad = 0.08 * max(xmax - xmin, ymax - ymin, 0.1)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()
    fig.savefig(out_dir / f"{safe_name}.png")
    fig.savefig(out_dir / f"{safe_name}.pdf")
    plt.close(fig)


# =============================================================================
# PROCESSING
# =============================================================================

def process_mode(
    mode: str,
    rows: List[Dict[str, str]],
    model: Any,
    ppo_cfg: CR3BPConfig,
    ppo_weights: RewardWeights,
    nom_cfg: CR3BPConfig,
    nom_weights: RewardWeights,
    ref: Dict[str, Any],
    out_dir: Path,
) -> Tuple[List[TrajectoryRecord], List[TrajectoryRecord], Path]:
    max_steps = MAX_STEPS_TLI if mode == "tli" else MAX_STEPS_MCC
    nominal_action = build_reference_action(mode, nom_cfg, ref)

    ppo_records: List[TrajectoryRecord] = []
    nom_records: List[TrajectoryRecord] = []

    sample_csv = out_dir / f"{mode}_validation_samples.csv"
    fieldnames = [
        "mode", "sample_idx", "sigma_pos_m", "sigma_vel_mps",
        "dx_m", "dy_m", "dvx_mps", "dvy_mps",
        "stored_ppo_pure_success", "rerun_ppo_success", "nominal_success",
        "rerun_ppo_reason", "nominal_reason",
        "rerun_ppo_steps", "nominal_steps", "rerun_ppo_clean_success_no_impact",
        "nominal_clean_success_no_impact",

        "rerun_ppo_audit_earth_impact_any",
        "nominal_audit_earth_impact_any",

        "rerun_ppo_audit_earth_impact_after_flyby",
        "nominal_audit_earth_impact_after_flyby",

        "rerun_ppo_audit_earth_impact_after_corridor",
        "nominal_audit_earth_impact_after_corridor",

        "rerun_ppo_audit_success_then_earth_impact",
        "nominal_audit_success_then_earth_impact",

        "rerun_ppo_audit_moon_impact_any",
        "nominal_audit_moon_impact_any",

        "rerun_ppo_audit_min_rE_nd",
        "nominal_audit_min_rE_nd",

        "rerun_ppo_audit_min_rM_nd",
        "nominal_audit_min_rM_nd",

        "rerun_ppo_audit_min_rE_postflyby_nd",
        "nominal_audit_min_rE_postflyby_nd",
    ]

    with open(sample_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            sample_idx = int(parse_float(row, "sample_idx", i))
            sp = parse_float(row, "sigma_pos_m")
            sv = parse_float(row, "sigma_vel_mps")
            dx = parse_float(row, "dx_m")
            dy = parse_float(row, "dy_m")
            dvx = parse_float(row, "dvx_mps")
            dvy = parse_float(row, "dvy_mps")

            env_ppo = make_env(copy.deepcopy(ppo_cfg), ppo_weights, SEED + i)
            reset_and_perturb(env_ppo, mode, ref, dx, dy, dvx, dvy)
            ppo_cls, ppo_traj = run_ppo_from_current_state(model, env_ppo, mode, max_steps)

            env_nom = make_env(copy.deepcopy(nom_cfg), nom_weights, SEED + i)
            reset_and_perturb(env_nom, mode, ref, dx, dy, dvx, dvy)
            nom_cls, nom_traj = run_nominal_episode(env_nom, mode, nominal_action, max_steps)

            ppo_records.append(TrajectoryRecord(
                mode=mode,
                controller="ppo",
                sample_idx=sample_idx,
                sigma_pos_m=sp,
                sigma_vel_mps=sv,
                success=bool(ppo_cls["clean_success_no_impact"]),
                reason=str(ppo_cls["reason"]),
                traj=ppo_traj,
            ))
            nom_records.append(TrajectoryRecord(
                mode=mode,
                controller="nominal",
                sample_idx=sample_idx,
                sigma_pos_m=sp,
                sigma_vel_mps=sv,
                success=bool(nom_cls["clean_success_no_impact"]),
                reason=str(nom_cls["reason"]),
                traj=nom_traj,
            ))

            writer.writerow({
                "mode": mode,
                "sample_idx": sample_idx,
                "sigma_pos_m": sp,
                "sigma_vel_mps": sv,
                "dx_m": dx,
                "dy_m": dy,
                "dvx_mps": dvx,
                "dvy_mps": dvy,
                "stored_ppo_pure_success": parse_bool(row.get("pure_success", False)),
                "rerun_ppo_success": bool(ppo_cls["success"]),
                "nominal_success": bool(nom_cls["success"]),
                "rerun_ppo_reason": str(ppo_cls["reason"]),
                "nominal_reason": str(nom_cls["reason"]),
                "rerun_ppo_steps": int(ppo_cls["n_steps"]),
                "nominal_steps": int(nom_cls["n_steps"]),
                "rerun_ppo_clean_success_no_impact": bool(ppo_cls["clean_success_no_impact"]),
                "nominal_clean_success_no_impact": bool(nom_cls["clean_success_no_impact"]),

                "rerun_ppo_audit_earth_impact_any": bool(ppo_cls["audit_earth_impact_any"]),
                "nominal_audit_earth_impact_any": bool(nom_cls["audit_earth_impact_any"]),

                "rerun_ppo_audit_earth_impact_after_flyby": bool(ppo_cls["audit_earth_impact_after_flyby"]),
                "nominal_audit_earth_impact_after_flyby": bool(nom_cls["audit_earth_impact_after_flyby"]),

                "rerun_ppo_audit_earth_impact_after_corridor": bool(ppo_cls["audit_earth_impact_after_corridor"]),
                "nominal_audit_earth_impact_after_corridor": bool(nom_cls["audit_earth_impact_after_corridor"]),

                "rerun_ppo_audit_success_then_earth_impact": bool(ppo_cls["audit_success_then_earth_impact"]),
                "nominal_audit_success_then_earth_impact": bool(nom_cls["audit_success_then_earth_impact"]),

                "rerun_ppo_audit_moon_impact_any": bool(ppo_cls["audit_moon_impact_any"]),
                "nominal_audit_moon_impact_any": bool(nom_cls["audit_moon_impact_any"]),

                "rerun_ppo_audit_min_rE_nd": float(ppo_cls["audit_min_rE_nd"]),
                "nominal_audit_min_rE_nd": float(nom_cls["audit_min_rE_nd"]),

                "rerun_ppo_audit_min_rM_nd": float(ppo_cls["audit_min_rM_nd"]),
                "nominal_audit_min_rM_nd": float(nom_cls["audit_min_rM_nd"]),

                "rerun_ppo_audit_min_rE_postflyby_nd": float(ppo_cls["audit_min_rE_postflyby_nd"]),
                "nominal_audit_min_rE_postflyby_nd": float(nom_cls["audit_min_rE_postflyby_nd"]),
            })

            if (i + 1) == 1 or (i + 1) % PRINT_EVERY == 0:
                print(
                    f"{mode.upper()} {i+1}/{len(rows)} | "
                    f"PPO SR={100*np.mean([r.success for r in ppo_records]):.1f}% | "
                    f"Nominal SR={100*np.mean([r.success for r in nom_records]):.1f}%"
                )

    return ppo_records, nom_records, sample_csv


def summarize_records(records: List[TrajectoryRecord]) -> Dict[str, Any]:
    if not records:
        return {"N": 0, "success_count": 0, "success_rate": float("nan")}
    return {
        "N": len(records),
        "success_count": int(sum(r.success for r in records)),
        "success_rate": float(np.mean([r.success for r in records])),
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_dir / "sensitivity analysis" / f"{RUN_TAG}_{timestamp_str()}")

    print("\n" + "=" * 78)
    print("PPO VS NOMINAL TRAJECTORY VALIDATION, FIXED")
    print("=" * 78)
    print(f"Script directory : {script_dir}")
    print(f"Output directory : {out_dir}")
    print(f"CR3BP V*         : {cr3bp_vstar_kms():.9f} km/s")

    tli_folder = resolve_path(script_dir, TLI_SENSITIVITY_FOLDER)
    mcc_folder = resolve_path(script_dir, MCC_SENSITIVITY_FOLDER)
    tli_ref_path = resolve_path(script_dir, TLI_REFERENCE_JSON)
    mcc_ref_path = resolve_path(script_dir, MCC_REFERENCE_JSON)

    tli_ref = load_json(tli_ref_path)
    mcc_ref = load_json(mcc_ref_path)

    tli_model_path = resolve_path(script_dir, PPO_TLI_MODEL_PATH) if PPO_TLI_MODEL_PATH else auto_find_policy(script_dir, AUTO_TLI_MODEL_KEYWORDS)
    mcc_model_path = resolve_path(script_dir, PPO_MCC_MODEL_PATH) if PPO_MCC_MODEL_PATH else auto_find_policy(script_dir, AUTO_MCC_MODEL_KEYWORDS)

    print(f"TLI samples       : {tli_folder}")
    print(f"MCC samples       : {mcc_folder}")
    print(f"TLI reference     : {tli_ref_path}")
    print(f"MCC reference     : {mcc_ref_path}")
    print(f"TLI PPO model     : {tli_model_path}")
    print(f"MCC PPO model     : {mcc_model_path}")

    print("\nLoading models and recovered configs...")
    tli_model = load_custom_model(tli_model_path)
    mcc_model = load_custom_model(mcc_model_path)

    tli_ppo_cfg, tli_weights, tli_recovered = build_cfg_and_weights_from_policy(tli_model_path)
    mcc_ppo_cfg, mcc_weights, mcc_recovered = build_cfg_and_weights_from_policy(mcc_model_path)

    repair_cfg_observation_space_to_model(tli_ppo_cfg, tli_model, "tli")
    repair_cfg_observation_space_to_model(mcc_ppo_cfg, mcc_model, "mcc")

    apply_eval_settings_to_ppo_cfg(tli_ppo_cfg, "tli", tli_ref, mcc_ref)
    apply_eval_settings_to_ppo_cfg(mcc_ppo_cfg, "mcc", tli_ref, mcc_ref)

    # Verify observation dimensions before expensive rerun.
    test_tli_env = make_env(copy.deepcopy(tli_ppo_cfg), tli_weights, SEED)
    test_mcc_env = make_env(copy.deepcopy(mcc_ppo_cfg), mcc_weights, SEED)

    print("\nObservation layout check:")
    print(f"  TLI model obs dim : {int(tli_model.observation_space.shape[0])}")
    print(f"  TLI env obs dim   : {int(test_tli_env.observation_space.shape[0])}")
    print(f"  TLI obs schema    : {get_obs_schema(test_tli_env)}")
    print(f"  MCC model obs dim : {int(mcc_model.observation_space.shape[0])}")
    print(f"  MCC env obs dim   : {int(test_mcc_env.observation_space.shape[0])}")
    print(f"  MCC obs schema    : {get_obs_schema(test_mcc_env)}")

    if int(test_tli_env.observation_space.shape[0]) != int(tli_model.observation_space.shape[0]):
        raise ValueError("TLI observation dimension mismatch after repair.")
    if int(test_mcc_env.observation_space.shape[0]) != int(mcc_model.observation_space.shape[0]):
        raise ValueError("MCC observation dimension mismatch after repair.")

    tli_nom_cfg = make_nominal_tli_cfg(tli_ref)
    mcc_nom_cfg = make_nominal_mcc_cfg(mcc_ref)
    nominal_weights = RewardWeights()

    tli_rows = read_sample_rows(tli_folder)
    mcc_rows = read_sample_rows(mcc_folder)

    print("\nRerunning TLI PPO + nominal...")
    tli_ppo_records, tli_nom_records, tli_csv = process_mode(
        "tli", tli_rows, tli_model, tli_ppo_cfg, tli_weights, tli_nom_cfg, nominal_weights, tli_ref, out_dir
    )

    print("\nRerunning MCC PPO + nominal...")
    mcc_ppo_records, mcc_nom_records, mcc_csv = process_mode(
        "mcc", mcc_rows, mcc_model, mcc_ppo_cfg, mcc_weights, mcc_nom_cfg, nominal_weights, mcc_ref, out_dir
    )

    print("\nSaving overlay plots...")
    plot_overlay(out_dir, "TLI PPO trajectory overlay", tli_ppo_cfg, tli_ppo_records)
    plot_overlay(out_dir, "TLI nominal trajectory overlay", tli_nom_cfg, tli_nom_records)
    plot_overlay(out_dir, "MCC PPO trajectory overlay", mcc_ppo_cfg, mcc_ppo_records)
    plot_overlay(out_dir, "MCC nominal trajectory overlay", mcc_nom_cfg, mcc_nom_records)

    meta = {
        "created": datetime.now().isoformat(),
        "run_tag": RUN_TAG,
        "tli_folder": str(tli_folder),
        "mcc_folder": str(mcc_folder),
        "tli_reference_json": str(tli_ref_path),
        "mcc_reference_json": str(mcc_ref_path),
        "tli_model_path": str(tli_model_path),
        "mcc_model_path": str(mcc_model_path),
        "tli_recovered": tli_recovered,
        "mcc_recovered": mcc_recovered,
        "summary": {
            "tli_ppo": summarize_records(tli_ppo_records),
            "tli_nominal": summarize_records(tli_nom_records),
            "mcc_ppo": summarize_records(mcc_ppo_records),
            "mcc_nominal": summarize_records(mcc_nom_records),
        },
        "outputs": {
            "tli_validation_csv": str(tli_csv),
            "mcc_validation_csv": str(mcc_csv),
            "plots": [
                str(out_dir / "tli_ppo_trajectory_overlay.png"),
                str(out_dir / "tli_nominal_trajectory_overlay.png"),
                str(out_dir / "mcc_ppo_trajectory_overlay.png"),
                str(out_dir / "mcc_nominal_trajectory_overlay.png"),
            ],
        },
    }
    save_json(out_dir / "validation_metadata.json", meta)

    summary_txt = out_dir / "validation_summary.txt"
    lines = ["PPO vs nominal validation summary", "=" * 60, ""]
    for k, v in meta["summary"].items():
        lines.append(f"{k}: N={v['N']}, success={v['success_count']}, SR={100*v['success_rate']:.1f}%")
    lines.append("")
    lines.append(f"TLI validation CSV: {tli_csv}")
    lines.append(f"MCC validation CSV: {mcc_csv}")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Output directory : {out_dir}")
    print(f"Summary          : {summary_txt}")
    print("Plots:")
    print(f"  {out_dir / 'tli_ppo_trajectory_overlay.png'}")
    print(f"  {out_dir / 'tli_nominal_trajectory_overlay.png'}")
    print(f"  {out_dir / 'mcc_ppo_trajectory_overlay.png'}")
    print(f"  {out_dir / 'mcc_nominal_trajectory_overlay.png'}")


if __name__ == "__main__":
    main()
