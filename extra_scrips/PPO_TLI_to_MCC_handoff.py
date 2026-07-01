"""
============================================================
PPO-A STAGED-TLI -> PPO-B MCC HANDOFF BUILDER
============================================================

Save as:
    PPO_TLI_to_MCC_handoff.py

Place in the same folder as:
    train_ppo_v4.py
    config.py
    cr3bp_env_v4.py
    cr3bp_plotting_v4.py

What this script does:
    1. Lets you select a PPO-A saved run.
    2. Lets you select a PPO-A saved policy.
    3. Rebuilds the PPO-A eval environment from the selected run folder's
       run_config.txt and saved curriculum stage.
    4. Runs the PPO-A policy deterministically ONLY UNTIL staged TLI commit.
    5. Uses env.tli_state_after_burn as the true post-multiple-burn TLI state.
    6. Optionally propagates ballistically after TLI, default 30 minutes.
    7. Saves that state as a PPO-B handoff-state scenario library.

Important:
    This script does NOT use the old PPO-B baseline single-TLI command logic.
    It saves a direct physical state_handoff, which PPO-B already supports.

Run:
    python PPO_TLI_to_MCC_handoff.py

Then in curriculum_ppob.py, set:
    MAIN_LIB = "rough_scenario_classification/<your_saved_file>.npz"
    MAIN_CASE_IDX = 0
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Console helpers
# ============================================================

def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        name = "ppob_handoff"
    name = re.sub(r"[^\w\-.]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def choose_from_list(items: List[Any], title: str, item_to_str=None) -> Any:
    if not items:
        raise FileNotFoundError(f"No selectable items for: {title}")

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    for i, item in enumerate(items):
        text = item_to_str(item) if item_to_str is not None else str(item)
        print(f"[{i:>3d}] {text}")

    while True:
        s = input("\nSelect index: ").strip()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(items):
                return items[idx]
        print("Invalid selection. Try again.")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    s = input(f"{prompt} {suffix}: ").strip().lower()
    if s == "":
        return bool(default)
    return s in ("y", "yes", "1", "true")


def ask_str(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(f"{prompt}: ").strip()
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else str(default)


def ask_float(prompt: str, default: float) -> float:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return float(default)
        try:
            return float(s)
        except Exception:
            print("Please enter a valid number.")


def ask_int(prompt: str, default: int) -> int:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return int(default)
        try:
            return int(s)
        except Exception:
            print("Please enter a valid integer.")


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
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


# ============================================================
# Project imports
# ============================================================

def import_project_modules():
    from config import (
        RUN,
        CR3BPConfig,
        RewardConfig,
        RewardWeights,
    )

    from cr3bp_env_v4 import (
        CR3BPFreeReturnEnv,
        SeanStyleReward,
        get_obs_schema,
        minutes_to_nondim_time,
        nondim_time_to_minutes,
        kms_to_nondim_dv,
        rk4_step,
    )

    from cr3bp_plotting_v4 import (
        plot_trajectory,
        plot_trajectory_earth_centered_inertial,
    )

    from train_ppo_v4 import (
        parse_run_config_txt,
        parse_saved_curriculum_stages,
        infer_saved_stage_index_from_step,
        apply_saved_stage_to_cfg_and_weights,
        extract_step_from_policy_name,
    )

    from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import (
        TimeAwareRecurrentPPOv2,
    )

    return {
        "RUN": RUN,
        "CR3BPConfig": CR3BPConfig,
        "RewardConfig": RewardConfig,
        "RewardWeights": RewardWeights,
        "CR3BPFreeReturnEnv": CR3BPFreeReturnEnv,
        "SeanStyleReward": SeanStyleReward,
        "get_obs_schema": get_obs_schema,
        "minutes_to_nondim_time": minutes_to_nondim_time,
        "nondim_time_to_minutes": nondim_time_to_minutes,
        "kms_to_nondim_dv": kms_to_nondim_dv,
        "rk4_step": rk4_step,
        "plot_trajectory": plot_trajectory,
        "plot_trajectory_earth_centered_inertial": plot_trajectory_earth_centered_inertial,
        "parse_run_config_txt": parse_run_config_txt,
        "parse_saved_curriculum_stages": parse_saved_curriculum_stages,
        "infer_saved_stage_index_from_step": infer_saved_stage_index_from_step,
        "apply_saved_stage_to_cfg_and_weights": apply_saved_stage_to_cfg_and_weights,
        "extract_step_from_policy_name": extract_step_from_policy_name,
        "TimeAwareRecurrentPPOv2": TimeAwareRecurrentPPOv2,
    }


# ============================================================
# Saved run and policy discovery
# ============================================================

def find_saved_root(script_dir: Path, RUN) -> Path:
    candidate = script_dir / str(RUN.saved_root_name)
    if candidate.exists():
        return candidate.resolve()

    s = ask_str("Path to your Saved Policies folder", str(candidate))
    root = Path(s).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Saved Policies folder not found:\n{root}")

    return root


def display_path_from_saved_root(path: Path, saved_root: Path) -> str:
    try:
        return str(path.relative_to(saved_root))
    except Exception:
        return str(path)


def list_run_dirs(saved_root: Path) -> List[Path]:
    runs: List[Path] = []

    for p in saved_root.iterdir():
        if not p.is_dir():
            continue

        has_run_config = (p / "run_config.txt").exists()
        has_zip = any(z.is_file() and z.suffix.lower() == ".zip" for z in p.rglob("*.zip"))
        has_plots = any(c.is_dir() and c.name.startswith("plots_") for c in p.iterdir())

        if has_run_config or has_zip or has_plots:
            runs.append(p)

    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def run_dir_looks_like_ppoa(run_dir: Path, mods) -> bool:
    parse_run_config_txt = mods["parse_run_config_txt"]

    saved = parse_run_config_txt(run_dir)
    name = run_dir.name.lower()
    trainer_mode = str(saved.get("trainer_mode", "")).lower()

    if trainer_mode == "ppo_a":
        return True

    if trainer_mode.startswith("ppo_b"):
        return False

    if "ppoa" in name or "ppo_a" in name or "tli" in name:
        return True

    return False


def list_ppoa_run_dirs(saved_root: Path, mods) -> List[Path]:
    return [p for p in list_run_dirs(saved_root) if run_dir_looks_like_ppoa(p, mods)]


def list_policy_files_in_run(run_dir: Path) -> List[Path]:
    files = [p for p in run_dir.rglob("*.zip") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


# ============================================================
# Config reconstruction
# ============================================================


def get_saved_optional_float(saved: Dict[str, Any], key: str) -> Optional[float]:
    """
    Read a float from parsed run_config values.

    Handles:
        None
        "None"
        ""
        numeric strings
        numeric values
    """
    if key not in saved:
        return None

    val = saved.get(key)

    if val is None:
        return None

    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() in ("none", "null"):
            return None
        try:
            return float(s)
        except Exception:
            return None

    try:
        return float(val)
    except Exception:
        return None


def apply_saved_run_level_physics_overrides(cfg, saved: Dict[str, Any], mods):
    """
    Re-apply RUN-level physical overrides saved in run_config.txt.

    Important bug fix:
        tli_dv_max_kms belongs to RUN, not CR3BPConfig.
        If we only rebuild CR3BPConfig fields, this value is ignored and
        cfg.dv_max_tli can stay at its old nondim default 4.4.

    For your PPO-A staged setup, tli_dv_max_kms = 0.40 should become:
        cfg.dv_max_tli = kms_to_nondim_dv(0.40)

    Same logic for MCC.
    """
    kms_to_nondim_dv = mods["kms_to_nondim_dv"]

    tli_dv_max_kms = get_saved_optional_float(saved, "tli_dv_max_kms")
    mcc_dv_max_kms = get_saved_optional_float(saved, "mcc_dv_max_kms")

    if tli_dv_max_kms is not None:
        cfg.dv_max_tli = float(kms_to_nondim_dv(tli_dv_max_kms))

    if mcc_dv_max_kms is not None:
        cfg.dv_max_mcc = float(kms_to_nondim_dv(mcc_dv_max_kms))

    return cfg


def repair_ppoa_staged_tli_physics(cfg, saved: Dict[str, Any], recovered: Dict[str, Any], mods):
    """
    Hard repair for PPO-A staged-TLI eval.

    Why this exists:
        Some saved run_config files contain the correct observation flags, but
        the base CR3BPConfig still has:
            staged_tli_cumulative_dv_target = 0.0
            staged_tli_max_burn_count = 12
            dv_max_tli = 4.4

        That creates exactly the failure you saw:
            - obs repair makes staged_tli_enabled=True
            - target remains 0.0
            - first burn instantly commits
            - if dv_max_tli remains 4.4, the first burn can be huge

    This repair restores the PPO-A staged training physics:
        - staged target: 3.1 km/s converted to nondim
        - per-step TLI cap: from saved tli_dv_max_kms, usually 0.40 km/s
        - max staged burns: 60 unless saved stage gives a larger value
        - min commit fraction: 1.0
    """
    kms_to_nondim_dv = mods["kms_to_nondim_dv"]

    trainer_mode = str(recovered.get("trainer_mode", getattr(cfg, "trainer_mode", ""))).lower()
    stage_name = str(recovered.get("stage_name", "")).lower()

    is_ppoa = trainer_mode == "ppo_a" or "ppo_a" in stage_name or "ppoa" in stage_name
    uses_staged_obs = bool(getattr(cfg, "add_staged_tli_obs", False))

    if not is_ppoa:
        return cfg

    # PPO-A staged policies with 12 obs need the staged-TLI mechanics too.
    if uses_staged_obs or bool(getattr(cfg, "staged_tli_enabled", False)):
        cfg.staged_tli_enabled = True
        cfg.staged_tli_commit_on_cumulative_dv = True
        cfg.staged_tli_limit_burn_count = True

        # If target is missing/zero, use the PPO-A curriculum value.
        # The uploaded PPO-A curriculum uses kms_to_nondim_dv(3.1).
        target = float(getattr(cfg, "staged_tli_cumulative_dv_target", 0.0) or 0.0)
        if target <= 1e-9:
            cfg.staged_tli_cumulative_dv_target = float(kms_to_nondim_dv(3.1))

        # If max count is still the base config default, restore PPO-A curriculum value.
        if int(getattr(cfg, "staged_tli_max_burn_count", 0) or 0) < 20:
            cfg.staged_tli_max_burn_count = 60

        cfg.staged_tli_min_commit_frac_of_target = 1.0

    # Always re-apply saved RUN-level dv caps after stage/config reconstruction.
    cfg = apply_saved_run_level_physics_overrides(cfg, saved, mods)

    # If run_config did not contain tli_dv_max_kms but this is PPO-A staged,
    # fall back to the current project default RUN value if available.
    RUN = mods["RUN"]
    if bool(getattr(cfg, "staged_tli_enabled", False)):
        saved_tli_cap = get_saved_optional_float(saved, "tli_dv_max_kms")
        if saved_tli_cap is None and getattr(RUN, "tli_dv_max_kms", None) is not None:
            cfg.dv_max_tli = float(kms_to_nondim_dv(float(RUN.tli_dv_max_kms)))

    return cfg


def print_physics_reconstruction_check(cfg):
    print("\nPPO-A staged physics check")
    print("-" * 70)
    print(f"dv_max_tli nondim                 : {getattr(cfg, 'dv_max_tli', None)}")
    print(f"dv_max_mcc nondim                 : {getattr(cfg, 'dv_max_mcc', None)}")
    print(f"staged_tli_enabled                : {getattr(cfg, 'staged_tli_enabled', None)}")
    print(f"staged_tli_commit_on_cumulative_dv: {getattr(cfg, 'staged_tli_commit_on_cumulative_dv', None)}")
    print(f"staged_tli_cumulative_dv_target   : {getattr(cfg, 'staged_tli_cumulative_dv_target', None)}")
    print(f"staged_tli_limit_burn_count       : {getattr(cfg, 'staged_tli_limit_burn_count', None)}")
    print(f"staged_tli_max_burn_count         : {getattr(cfg, 'staged_tli_max_burn_count', None)}")
    print(f"staged_tli_min_commit_frac        : {getattr(cfg, 'staged_tli_min_commit_frac_of_target', None)}")


def build_cfg_and_weights_from_policy_with_run_dir(
    policy_path: Path,
    run_dir: Path,
    mods,
):
    """
    Rebuild config from the selected RUN FOLDER, not from policy_path.parent.

    This matters because saved policies can be inside checkpoint/milestone
    subfolders. If we read from the wrong folder, the PPO-A observation space
    can be rebuilt incorrectly.
    """
    CR3BPConfig = mods["CR3BPConfig"]
    RewardWeights = mods["RewardWeights"]
    parse_run_config_txt = mods["parse_run_config_txt"]
    parse_saved_curriculum_stages = mods["parse_saved_curriculum_stages"]
    infer_saved_stage_index_from_step = mods["infer_saved_stage_index_from_step"]
    apply_saved_stage_to_cfg_and_weights = mods["apply_saved_stage_to_cfg_and_weights"]
    extract_step_from_policy_name = mods["extract_step_from_policy_name"]

    saved = parse_run_config_txt(run_dir)

    if len(saved) == 0:
        raise FileNotFoundError(
            "Could not read run_config.txt from selected run folder:\n"
            f"{run_dir}\n"
        )

    cfg = CR3BPConfig()

    for field_name in cfg.__dataclass_fields__.keys():
        if field_name in saved:
            setattr(cfg, field_name, saved[field_name])

    default_w = RewardWeights()
    weights = RewardWeights(
        w_flyby=float(saved.get("w_flyby", default_w.w_flyby)),
        w_velocity=float(saved.get("w_velocity", default_w.w_velocity)),
        w_dv=float(saved.get("w_dv", default_w.w_dv)),
        w_return=float(saved.get("w_return", default_w.w_return)),
        w_budget=float(saved.get("w_budget", default_w.w_budget)),
        w_escape=float(saved.get("w_escape", default_w.w_escape)),
        w_earth_crash=float(saved.get("w_earth_crash", default_w.w_earth_crash)),
        w_moon_crash=float(saved.get("w_moon_crash", default_w.w_moon_crash)),
        w_postflyby_earth_crash=float(
            saved.get("w_postflyby_earth_crash", default_w.w_postflyby_earth_crash)
        ),
        w_invalid_preflyby_earth_return=float(
            saved.get(
                "w_invalid_preflyby_earth_return",
                default_w.w_invalid_preflyby_earth_return,
            )
        ),
    )

    step_count = mods["extract_step_from_policy_name"](policy_path)
    saved_stages = parse_saved_curriculum_stages(run_dir)

    chosen_stage_idx = -1
    chosen_stage_name = "from_saved_run_config"

    if len(saved_stages) > 0:
        chosen_stage_idx = infer_saved_stage_index_from_step(saved_stages, step_count)
        chosen_stage = saved_stages[chosen_stage_idx]
        cfg, weights = apply_saved_stage_to_cfg_and_weights(cfg, weights, chosen_stage)
        chosen_stage_name = str(chosen_stage.get("stage_name", f"stage_{chosen_stage_idx + 1}"))

    # Re-apply RUN-level physical overrides like tli_dv_max_kms/mcc_dv_max_kms.
    cfg = apply_saved_run_level_physics_overrides(cfg, saved, mods)

    recovered = {
        "policy_step": int(step_count),
        "stage_idx": int(chosen_stage_idx),
        "stage_name": chosen_stage_name,
        "mcc_enabled": bool(getattr(cfg, "mcc_enabled", False)),
        "tli_only_mode": bool(getattr(cfg, "tli_only_mode", False)),
        "reward_after_tli_ballistic_enabled": bool(
            getattr(cfg, "reward_after_tli_ballistic_enabled", False)
        ),
        "config_file_found": (run_dir / "run_config.txt").exists(),
        "trainer_mode": str(getattr(cfg, "trainer_mode", "ppo_a")),
        "tli_control_mode": str(getattr(cfg, "tli_control_mode", "full")),
        "run_dir_used_for_config": str(run_dir),
    }

    return cfg, weights, recovered


def make_env(cfg, weights, mods):
    RUN = mods["RUN"]
    RewardConfig = mods["RewardConfig"]
    CR3BPFreeReturnEnv = mods["CR3BPFreeReturnEnv"]
    SeanStyleReward = mods["SeanStyleReward"]

    env = CR3BPFreeReturnEnv(
        cfg,
        seed=RUN.eval_seed,
        reward_model=SeanStyleReward(RewardConfig(), weights),
    )
    env.set_debug_eval(True)
    return env


def try_repair_obs_space_for_ppoa(cfg, weights, model, mods):
    """
    Last-resort repair if an older run_config did not save every observation flag.
    This is still PPO-A only. It does not use PPO-B.
    """
    expected_shape = tuple(model.observation_space.shape)

    candidates = [
        {
            "add_phase_angle_obs": True,
            "add_mode_obs": True,
            "add_legacy_mode_obs": False,
            "add_staged_tli_obs": True,
            "staged_tli_enabled": True,
            "reason": "modern PPO-A staged TLI",
        },
        {
            "add_phase_angle_obs": True,
            "add_mode_obs": True,
            "add_legacy_mode_obs": True,
            "add_staged_tli_obs": True,
            "staged_tli_enabled": True,
            "reason": "legacy plus staged TLI",
        },
        {
            "add_phase_angle_obs": True,
            "add_mode_obs": True,
            "add_legacy_mode_obs": False,
            "add_staged_tli_obs": False,
            "staged_tli_enabled": False,
            "reason": "phase only",
        },
        {
            "add_phase_angle_obs": True,
            "add_mode_obs": True,
            "add_legacy_mode_obs": True,
            "add_staged_tli_obs": False,
            "staged_tli_enabled": False,
            "reason": "legacy mode only",
        },
        {
            "add_phase_angle_obs": False,
            "add_mode_obs": False,
            "add_legacy_mode_obs": False,
            "add_staged_tli_obs": False,
            "staged_tli_enabled": False,
            "reason": "base only",
        },
    ]

    for cand in candidates:
        test_cfg = type(cfg)(**vars(cfg))

        for key, val in cand.items():
            if key == "reason":
                continue
            if hasattr(test_cfg, key):
                setattr(test_cfg, key, val)

        test_env = make_env(test_cfg, weights, mods)

        if tuple(test_env.observation_space.shape) == expected_shape:
            print("\n[OBS REPAIR] Applied compatibility repair")
            print("-" * 70)
            print(f"reason              : {cand['reason']}")
            print(f"model obs           : {model.observation_space.shape}")
            print(f"env obs             : {test_env.observation_space.shape}")
            print(f"add_phase_angle_obs : {getattr(test_cfg, 'add_phase_angle_obs', None)}")
            print(f"add_mode_obs        : {getattr(test_cfg, 'add_mode_obs', None)}")
            print(f"add_legacy_mode_obs : {getattr(test_cfg, 'add_legacy_mode_obs', None)}")
            print(f"add_staged_tli_obs  : {getattr(test_cfg, 'add_staged_tli_obs', None)}")
            print(f"staged_tli_enabled  : {getattr(test_cfg, 'staged_tli_enabled', None)}")
            return test_cfg, test_env

    return cfg, None


def build_env_and_model(policy_path: Path, run_dir: Path, mods):
    RUN = mods["RUN"]
    TimeAwareRecurrentPPOv2 = mods["TimeAwareRecurrentPPOv2"]
    get_obs_schema = mods["get_obs_schema"]

    cfg, weights, recovered = build_cfg_and_weights_from_policy_with_run_dir(
        policy_path=policy_path,
        run_dir=run_dir,
        mods=mods,
    )

    trainer_mode = str(recovered.get("trainer_mode", "")).lower()
    if trainer_mode != "ppo_a":
        raise RuntimeError(
            "Selected policy did not recover as PPO-A.\n"
            f"Recovered trainer_mode = {trainer_mode}\n"
        )

    # Repair PPO-A staged physics before creating the env.
    cfg = repair_ppoa_staged_tli_physics(cfg, {}, recovered, mods)

    model = TimeAwareRecurrentPPOv2.load(str(policy_path), device=RUN.device)
    env = make_env(cfg, weights, mods)

    print_physics_reconstruction_check(cfg)

    print("\nObservation-space check")
    print("-" * 70)
    print(f"model expects       : {model.observation_space.shape}")
    print(f"env gives           : {env.observation_space.shape}")
    print(f"add_phase_angle_obs : {getattr(cfg, 'add_phase_angle_obs', None)}")
    print(f"add_mode_obs        : {getattr(cfg, 'add_mode_obs', None)}")
    print(f"add_legacy_mode_obs : {getattr(cfg, 'add_legacy_mode_obs', None)}")
    print(f"add_staged_tli_obs  : {getattr(cfg, 'add_staged_tli_obs', None)}")
    print(f"staged_tli_enabled  : {getattr(cfg, 'staged_tli_enabled', None)}")
    print(f"obs schema          : {get_obs_schema(env)}")

    if tuple(env.observation_space.shape) != tuple(model.observation_space.shape):
        print("\n[WARN] Observation mismatch after run_config reconstruction.")
        print("[WARN] Trying PPO-A observation compatibility repair...")

        cfg_repaired, env_repaired = try_repair_obs_space_for_ppoa(
            cfg=cfg,
            weights=weights,
            model=model,
            mods=mods,
        )

        if env_repaired is None:
            raise RuntimeError(
                "Observation mismatch could not be repaired.\n"
                f"Model expects: {model.observation_space.shape}\n"
                f"Env gives    : {env.observation_space.shape}\n"
            )

        cfg = repair_ppoa_staged_tli_physics(cfg_repaired, {}, recovered, mods)
        env = make_env(cfg, weights, mods)
        print_physics_reconstruction_check(cfg)

    return cfg, weights, recovered, env, model


# ============================================================
# Staged-TLI handoff rollout
# ============================================================

TLI_KIND_SET = {
    "TLI",
    "TLI_FINAL_BURN",
    "TLI_WAIT_COMMIT",
}


def is_tli_committed(env) -> bool:
    if getattr(env, "tli_state_after_burn", None) is not None:
        return True

    events = getattr(env, "burn_events", []) or []
    for ev in events:
        if str(ev.get("kind", "")) in TLI_KIND_SET:
            return True

    hist = getattr(env, "action_history", []) or []
    for row in hist:
        if str(row.get("burn_kind", "")) in TLI_KIND_SET:
            return True

    return False


def run_until_staged_tli_commit(env, model, max_policy_steps: int = 10000):
    """
    Run PPO-A deterministically only until TLI commit.

    This is deliberately different from a full PPO-A eval rollout:
    - full PPO-A eval usually continues into ballistic reward evaluation
    - this script stops at the handoff boundary
    """
    obs, info = env.reset()

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    rewards = []
    infos = []
    actions = []

    for step_idx in range(int(max_policy_steps)):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(info)
        actions.append(np.asarray(action, dtype=np.float64).copy())

        episode_start = np.array([terminated or truncated], dtype=bool)

        if is_tli_committed(env):
            return {
                "committed": True,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "step_idx": int(step_idx),
                "last_info": info,
                "rewards": np.asarray(rewards, dtype=np.float64),
                "actions": actions,
                "infos": infos,
            }

        if terminated or truncated:
            return {
                "committed": False,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "step_idx": int(step_idx),
                "last_info": info,
                "rewards": np.asarray(rewards, dtype=np.float64),
                "actions": actions,
                "infos": infos,
            }

    return {
        "committed": False,
        "terminated": False,
        "truncated": True,
        "step_idx": int(max_policy_steps),
        "last_info": infos[-1] if infos else {},
        "rewards": np.asarray(rewards, dtype=np.float64),
        "actions": actions,
        "infos": infos,
    }


def classify_burn_events(env) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ev in getattr(env, "burn_events", []) or []:
        kind = str(ev.get("kind", "UNKNOWN"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def print_staged_tli_diagnostics(env, rollout_result):
    counts = classify_burn_events(env)
    events = getattr(env, "burn_events", []) or []
    hist = getattr(env, "action_history", []) or []

    print("\nStaged-TLI diagnostics")
    print("-" * 70)
    print(f"committed                 : {rollout_result['committed']}")
    print(f"policy steps to handoff   : {rollout_result['step_idx'] + 1}")
    print(f"terminated at same step   : {rollout_result['terminated']}")
    print(f"truncated at same step    : {rollout_result['truncated']}")
    print(f"env.pre_tli_burn_count    : {getattr(env, 'pre_tli_burn_count', None)}")
    print(f"env.pre_tli_cum_dv        : {getattr(env, 'pre_tli_cum_dv', None)}")
    print(f"env.dv0                   : {getattr(env, 'dv0', None)}")
    print(f"burn event counts         : {counts}")
    print(f"action history rows       : {len(hist)}")
    print(f"burn events stored        : {len(events)}")

    print("\nBurn-event summary")
    print("-" * 70)
    for i, ev in enumerate(events):
        kind = str(ev.get("kind", "UNKNOWN"))
        t = ev.get("time", np.nan)
        dv = ev.get("dv_mag", ev.get("dv", np.nan))
        cum = ev.get("pre_tli_cum_dv", np.nan)
        bc = ev.get("pre_tli_burn_count", np.nan)
        print(f"[{i:03d}] kind={kind:18s} t={float_or_nan(t): .9f} dv={float_or_nan(dv): .9f} cum={float_or_nan(cum): .9f} count={bc}")

    pre_count = counts.get("PRE_TLI_BURN", 0)
    final_count = sum(counts.get(k, 0) for k in TLI_KIND_SET)

    if pre_count <= 1 and final_count <= 1:
        print("\n[WARNING]")
        print("It looks like the rollout only produced one/few pre-TLI burn events.")
        print("If this policy was trained for many staged burns, the eval config may still be wrong,")
        print("or the deterministic policy genuinely commits in one large final burn.")
        print("Check the observation-space printout and the burn-event summary above.")
    else:
        print("\n[OK]")
        print("Multiple pre-TLI burn events were detected before the staged TLI handoff.")


def float_or_nan(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def find_tli_time_from_action_history(env) -> Optional[float]:
    hist = getattr(env, "action_history", []) or []

    for row in hist:
        kind = str(row.get("burn_kind", ""))
        if kind in TLI_KIND_SET:
            # For handoff, the after-step time is normally the most useful.
            for key in ("time_after", "time_before"):
                try:
                    t = float(row.get(key, np.nan))
                    if np.isfinite(t):
                        return t
                except Exception:
                    pass

    return None


def find_tli_time_from_burn_events(env) -> Optional[float]:
    events = getattr(env, "burn_events", []) or []

    for ev in events:
        kind = str(ev.get("kind", ""))
        if kind in TLI_KIND_SET:
            try:
                t = float(ev.get("time", np.nan))
                if np.isfinite(t):
                    return t
            except Exception:
                pass

    return None


def get_tli_state_and_time(env) -> Tuple[np.ndarray, float, str]:
    state_after_tli = getattr(env, "tli_state_after_burn", None)

    if state_after_tli is None:
        raise RuntimeError(
            "No env.tli_state_after_burn was found.\n"
            "The deterministic PPO-A rollout did not commit staged TLI.\n"
        )

    state_after_tli = np.asarray(state_after_tli, dtype=np.float64).reshape(-1)

    if state_after_tli.size != 4:
        raise RuntimeError(
            f"Expected tli_state_after_burn shape (4,), got {state_after_tli.shape}"
        )

    t_tli = find_tli_time_from_action_history(env)
    source = "action_history"

    if t_tli is None:
        t_tli = find_tli_time_from_burn_events(env)
        source = "burn_events"

    if t_tli is None:
        t_tli = float(getattr(env, "t", 0.0))
        source = "fallback_env_time"

    return state_after_tli.copy(), float(t_tli), source


def propagate_ballistic_from_tli(
    state_after_tli: np.ndarray,
    t_tli: float,
    post_tli_minutes: float,
    cfg,
    mods,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Ballistically propagate the already-committed staged-TLI state.

    This is NOT applying a new burn.
    """
    minutes_to_nondim_time = mods["minutes_to_nondim_time"]
    rk4_step = mods["rk4_step"]

    dt_total = float(minutes_to_nondim_time(post_tli_minutes))

    if post_tli_minutes <= 0.0:
        s = np.asarray(state_after_tli, dtype=np.float64).copy()
        return (
            s.copy(),
            float(t_tli),
            s.reshape(1, 4),
            np.asarray([float(t_tli)], dtype=np.float64),
        )

    # 1-minute-ish output spacing, capped for speed.
    n_steps = max(1, int(math.ceil(post_tli_minutes)))
    n_steps = min(n_steps, 1000)
    dt = dt_total / float(n_steps)

    s = np.asarray(state_after_tli, dtype=np.float64).copy()
    t = float(t_tli)

    traj = [s.copy()]
    t_hist = [t]

    for _ in range(n_steps):
        s = rk4_step(float(cfg.mu), s, dt)
        t += dt
        traj.append(s.copy())
        t_hist.append(float(t))

    return (
        s.copy(),
        float(t),
        np.asarray(traj, dtype=np.float64),
        np.asarray(t_hist, dtype=np.float64),
    )


# ============================================================
# Plotting and reporting
# ============================================================

def plot_handoff_debug(
    cfg,
    env,
    post_tli_traj: np.ndarray,
    post_tli_t_hist: np.ndarray,
    mods,
    out_dir: Path,
    policy_path: Path,
):
    plot_trajectory = mods["plot_trajectory"]
    plot_trajectory_earth_centered_inertial = mods["plot_trajectory_earth_centered_inertial"]

    env_traj = np.asarray(getattr(env, "traj", []), dtype=np.float64)
    env_t_hist = np.asarray(getattr(env, "t_hist", []), dtype=np.float64)

    if env_traj.ndim != 2 or env_traj.shape[1] != 4:
        env_traj = np.zeros((0, 4), dtype=np.float64)

    if env_t_hist.ndim != 1 or len(env_t_hist) != len(env_traj):
        env_t_hist = np.zeros((len(env_traj),), dtype=np.float64)

    # Stitch pre-TLI rollout and post-TLI handoff coast for plotting only.
    if len(env_traj) > 0 and len(post_tli_traj) > 0:
        stitched_traj = np.vstack([env_traj, post_tli_traj])
        stitched_t = np.concatenate([env_t_hist, post_tli_t_hist])
    else:
        stitched_traj = post_tli_traj
        stitched_t = post_tli_t_hist

    burns = np.asarray(getattr(env, "burns", []), dtype=np.float64)
    if burns.ndim != 2 or burns.shape[0] == 0:
        burns = None

    rot_path = out_dir / "handoff_staged_tli_plus_post_tli_coast_rotating.png"
    plot_trajectory(
        cfg,
        stitched_traj,
        burns=burns,
        burn_events=getattr(env, "burn_events", None),
        ballistic_ref_traj=getattr(env, "ballistic_ref_traj", None),
        ballistic_terminal_marker=getattr(env, "ballistic_terminal_marker_rot", None),
        terminal_marker=getattr(env, "terminal_marker_rot", None),
        title=f"Staged PPO-A handoff: {policy_path.stem}",
        out_path=str(rot_path),
    )

    inert_path = out_dir / "handoff_staged_tli_plus_post_tli_coast_inertial.png"
    plot_trajectory_earth_centered_inertial(
        cfg,
        stitched_traj,
        stitched_t,
        ballistic_ref_traj=getattr(env, "ballistic_ref_traj", None),
        ballistic_ref_t_hist=getattr(env, "ballistic_ref_t_hist", None),
        title=f"Staged PPO-A handoff inertial: {policy_path.stem}",
        out_path=str(inert_path),
    )

    print("\nPlots saved")
    print("-" * 70)
    print(f"rotating : {rot_path}")
    print(f"inertial : {inert_path}")

    return {
        "rotating": rot_path,
        "inertial": inert_path,
    }


def save_debug_json(
    out_dir: Path,
    policy_path: Path,
    run_dir: Path,
    recovered: Dict[str, Any],
    env,
    rollout_result: Dict[str, Any],
    state_after_tli: np.ndarray,
    state_handoff: np.ndarray,
    t_tli: float,
    t_handoff: float,
    post_tli_minutes: float,
    npz_path: Path,
):
    payload = {
        "created_at": timestamp_str(),
        "policy_path": str(policy_path),
        "run_dir": str(run_dir),
        "recovered": make_jsonable(recovered),
        "rollout_result": {
            "committed": bool(rollout_result.get("committed", False)),
            "terminated": bool(rollout_result.get("terminated", False)),
            "truncated": bool(rollout_result.get("truncated", False)),
            "step_idx": int(rollout_result.get("step_idx", -1)),
            "last_info": make_jsonable(rollout_result.get("last_info", {})),
        },
        "staged_tli": {
            "pre_tli_burn_count": make_jsonable(getattr(env, "pre_tli_burn_count", None)),
            "pre_tli_cum_dv": make_jsonable(getattr(env, "pre_tli_cum_dv", None)),
            "dv0": make_jsonable(getattr(env, "dv0", None)),
            "burn_event_counts": classify_burn_events(env),
            "burn_events": make_jsonable(getattr(env, "burn_events", []) or []),
            "action_history": make_jsonable(getattr(env, "action_history", []) or []),
        },
        "handoff": {
            "state_after_tli": make_jsonable(state_after_tli),
            "state_handoff": make_jsonable(state_handoff),
            "t_tli": float(t_tli),
            "t_handoff": float(t_handoff),
            "post_tli_minutes": float(post_tli_minutes),
            "npz_path": str(npz_path),
        },
    }

    path = out_dir / "handoff_debug_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Debug JSON saved: {path}")
    return path


# ============================================================
# PPO-B handoff library saving
# ============================================================

def label_name(label: int) -> str:
    return {
        0: "good",
        1: "savable",
        2: "bad",
    }.get(int(label), "unknown")


def ask_case_label(default: int = 1) -> int:
    print("\nPPO-B case label")
    print("-" * 70)
    print("[0] good")
    print("[1] savable")
    print("[2] bad")

    while True:
        label = ask_int("Select label", default)
        if label in (0, 1, 2):
            return int(label)
        print("Please choose 0, 1, or 2.")


def save_handoff_npz(
    out_path: Path,
    policy_path: Path,
    run_dir: Path,
    recovered: Dict[str, Any],
    state_handoff: np.ndarray,
    t_handoff: float,
    state_after_tli: np.ndarray,
    t_tli: float,
    tli_time_source: str,
    post_tli_minutes: float,
    label: int,
    env,
    rollout_result: Dict[str, Any],
    post_tli_traj: np.ndarray,
    post_tli_t_hist: np.ndarray,
) -> Path:
    out_path = Path(out_path).resolve()
    ensure_dir(out_path.parent)

    state_handoff = np.asarray(state_handoff, dtype=np.float64).reshape(1, 4)
    state_after_tli = np.asarray(state_after_tli, dtype=np.float64).reshape(1, 4)

    # Keep metadata finite where PPO-B may read it.
    dv0 = float_or_nan(getattr(env, "dv0", 0.0))
    if not np.isfinite(dv0):
        dv0 = 0.0

    np.savez_compressed(
        out_path,

        # REQUIRED by PPO-B handoff-state library
        state_handoff=state_handoff,
        label=np.asarray([int(label)], dtype=np.int32),

        # Important optional metadata read by PPO-B if present
        t_handoff=np.asarray([float(t_handoff)], dtype=np.float64),
        state_after_tli=state_after_tli,
        t_after_tli=np.asarray([float(t_handoff - t_tli)], dtype=np.float64),
        dv0=np.asarray([float(dv0)], dtype=np.float64),

        # Useful metadata
        post_tli_minutes=np.asarray([float(post_tli_minutes)], dtype=np.float64),
        t_tli=np.asarray([float(t_tli)], dtype=np.float64),
        source_policy=np.asarray([str(policy_path.name)]),
        source_policy_path=np.asarray([str(policy_path.resolve())]),
        source_run_dir=np.asarray([str(run_dir.resolve())]),
        source_policy_step=np.asarray(
            [int(recovered.get("policy_step", -1))],
            dtype=np.int64,
        ),
        source_stage_idx=np.asarray(
            [int(recovered.get("stage_idx", -1))],
            dtype=np.int64,
        ),
        source_stage_name=np.asarray([str(recovered.get("stage_name", ""))]),
        source_trainer_mode=np.asarray([str(recovered.get("trainer_mode", ""))]),
        tli_time_source=np.asarray([str(tli_time_source)]),

        # Staged-TLI diagnostics
        pre_tli_burn_count=np.asarray(
            [int(getattr(env, "pre_tli_burn_count", 0) or 0)],
            dtype=np.int32,
        ),
        pre_tli_cum_dv=np.asarray(
            [float_or_nan(getattr(env, "pre_tli_cum_dv", 0.0))],
            dtype=np.float64,
        ),

        # Last rollout info
        rollout_committed=np.asarray(
            [1 if bool(rollout_result.get("committed", False)) else 0],
            dtype=np.int32,
        ),
        rollout_step_idx=np.asarray(
            [int(rollout_result.get("step_idx", -1))],
            dtype=np.int32,
        ),

        # Debug trajectory from immediate post-TLI to handoff
        post_tli_traj=post_tli_traj,
        post_tli_t_hist=post_tli_t_hist,

        # Human-readable schemas
        state_schema=np.asarray(["x", "y", "vx", "vy"]),
        label_schema=np.asarray(["0=good", "1=savable", "2=bad"]),
        library_note=np.asarray([
            "handoff-state library generated from staged PPO-A tli_state_after_burn"
        ]),
    )

    return out_path


def save_metadata_json(
    npz_path: Path,
    policy_path: Path,
    run_dir: Path,
    recovered: Dict[str, Any],
    post_tli_minutes: float,
    label: int,
    t_tli: float,
    t_handoff: float,
    tli_time_source: str,
    env,
    rollout_result: Dict[str, Any],
) -> Path:
    json_path = npz_path.with_suffix(".json")

    meta = {
        "created_at": timestamp_str(),
        "format": "ppo_b_handoff_state_library",
        "npz_path": str(npz_path),
        "source_policy": str(policy_path),
        "source_run_dir": str(run_dir),
        "recovered": make_jsonable(recovered),
        "post_tli_minutes": float(post_tli_minutes),
        "label": int(label),
        "label_name": label_name(label),
        "t_tli": float(t_tli),
        "t_handoff": float(t_handoff),
        "tli_time_source": str(tli_time_source),
        "staged_tli": {
            "pre_tli_burn_count": make_jsonable(getattr(env, "pre_tli_burn_count", None)),
            "pre_tli_cum_dv": make_jsonable(getattr(env, "pre_tli_cum_dv", None)),
            "dv0": make_jsonable(getattr(env, "dv0", None)),
            "burn_event_counts": classify_burn_events(env),
        },
        "how_to_use_in_curriculum_ppob": {
            "MAIN_LIB": str(npz_path),
            "MAIN_CASE_IDX": 0,
            "ppo_b_case_source": "scenario_library",
            "ppo_b_use_fixed_index": True,
            "ppo_b_fixed_index": 0,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return json_path


def validate_saved_handoff_library(npz_path: Path) -> bool:
    data = np.load(npz_path, allow_pickle=False)

    required = ["state_handoff", "label"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"[VALIDATION] Missing required keys: {missing}")
        return False

    state_handoff = np.asarray(data["state_handoff"], dtype=np.float64)
    label = np.asarray(data["label"], dtype=np.int32).reshape(-1)

    ok = True

    if state_handoff.ndim != 2 or state_handoff.shape[1] != 4:
        print(f"[VALIDATION] Bad state_handoff shape: {state_handoff.shape}")
        ok = False

    if len(label) != state_handoff.shape[0]:
        print("[VALIDATION] label length does not match state_handoff rows.")
        ok = False

    if "t_handoff" not in data:
        print("[VALIDATION] Warning: t_handoff missing. PPO-B usually expects it.")
        ok = False

    if not np.all(np.isfinite(state_handoff)):
        print("[VALIDATION] state_handoff contains non-finite values.")
        ok = False

    if ok:
        print("[VALIDATION] Saved .npz is compatible with PPO-B handoff-state library.")
        print(f"[VALIDATION] state_handoff shape: {state_handoff.shape}")
        print(f"[VALIDATION] label: {label.tolist()}")
        print(f"[VALIDATION] t_handoff: {np.asarray(data['t_handoff']).tolist()}")

    return ok


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 90)
    print("PPO-A staged-TLI -> PPO-B MCC handoff builder")
    print("=" * 90)

    script_dir = Path(__file__).resolve().parent
    mods = import_project_modules()

    RUN = mods["RUN"]
    nondim_time_to_minutes = mods["nondim_time_to_minutes"]

    saved_root = find_saved_root(script_dir, RUN)

    ppoa_runs = list_ppoa_run_dirs(saved_root, mods)
    if len(ppoa_runs) == 0:
        raise FileNotFoundError(
            "No PPO-A run folders found.\n"
            f"Saved root searched:\n{saved_root}"
        )

    run_dir = choose_from_list(
        ppoa_runs,
        title="Select PPO-A run folder",
        item_to_str=lambda p: display_path_from_saved_root(p, saved_root),
    )

    run_config_path = run_dir / "run_config.txt"

    print("\nSelected run")
    print("-" * 70)
    print(f"run_dir       : {display_path_from_saved_root(run_dir, saved_root)}")
    print(f"run_config.txt: {run_config_path if run_config_path.exists() else 'NOT FOUND'}")

    policies = list_policy_files_in_run(run_dir)
    if len(policies) == 0:
        raise FileNotFoundError(f"No .zip policies found inside:\n{run_dir}")

    policy_path = choose_from_list(
        policies,
        title="Select saved PPO-A policy (.zip)",
        item_to_str=lambda p: display_path_from_saved_root(p, saved_root),
    )

    print("\nLoading recovered PPO-A configuration and model...")
    cfg, weights, recovered, env, model = build_env_and_model(
        policy_path=policy_path,
        run_dir=run_dir,
        mods=mods,
    )

    print("\nRecovered policy config")
    print("-" * 70)
    print(f"policy file      : {policy_path.name}")
    print(f"policy step      : {recovered.get('policy_step', 'unknown')}")
    print(f"trainer_mode     : {recovered.get('trainer_mode', 'unknown')}")
    print(f"tli_control_mode : {recovered.get('tli_control_mode', 'unknown')}")
    print(f"inferred stage   : {recovered.get('stage_name', 'unknown')}")
    print(f"config file found: {recovered.get('config_file_found', False)}")
    print(f"staged enabled   : {getattr(cfg, 'staged_tli_enabled', None)}")
    print(f"staged target    : {getattr(cfg, 'staged_tli_cumulative_dv_target', None)}")
    print(f"max burn count   : {getattr(cfg, 'staged_tli_max_burn_count', None)}")

    out_dir = ensure_dir(run_dir / f"staged_tli_handoff_{timestamp_str()}")

    print("\nRunning deterministic PPO-A only until staged TLI commit...")
    rollout_result = run_until_staged_tli_commit(
        env=env,
        model=model,
        max_policy_steps=ask_int("Max PPO-A policy steps before giving up", 10000),
    )

    print_staged_tli_diagnostics(env, rollout_result)

    if not bool(rollout_result["committed"]):
        save_debug_json(
            out_dir=out_dir,
            policy_path=policy_path,
            run_dir=run_dir,
            recovered=recovered,
            env=env,
            rollout_result=rollout_result,
            state_after_tli=np.full(4, np.nan),
            state_handoff=np.full(4, np.nan),
            t_tli=np.nan,
            t_handoff=np.nan,
            post_tli_minutes=np.nan,
            npz_path=out_dir / "NO_HANDOFF_SAVED.npz",
        )
        raise RuntimeError(
            "No staged TLI commit occurred, so no PPO-B handoff was saved.\n"
            f"Debug output folder:\n{out_dir}"
        )

    state_after_tli, t_tli, tli_time_source = get_tli_state_and_time(env)

    print("\nDetected staged-TLI handoff boundary")
    print("-" * 70)
    print(f"tli time source      : {tli_time_source}")
    print(f"t_tli nondim         : {t_tli:.12f}")
    print(f"t_tli minutes        : {nondim_time_to_minutes(t_tli):.6f}")
    print(f"state_after_tli      : {state_after_tli}")

    print("\nHandoff timing")
    print("-" * 70)
    print("0 min means PPO-B starts immediately after the final staged TLI commit.")
    print("30 min means PPO-B starts after a ballistic coast from that committed state.")
    post_tli_minutes = ask_float("Minutes after TLI for PPO-B handoff", 30.0)

    if post_tli_minutes < 0.0:
        raise ValueError("post_tli_minutes must be >= 0.")

    label = ask_case_label(default=1)

    state_handoff, t_handoff, post_tli_traj, post_tli_t_hist = propagate_ballistic_from_tli(
        state_after_tli=state_after_tli,
        t_tli=t_tli,
        post_tli_minutes=post_tli_minutes,
        cfg=cfg,
        mods=mods,
    )

    print("\nGenerated PPO-B handoff state")
    print("-" * 70)
    print(f"post TLI minutes : {post_tli_minutes:.6f}")
    print(f"t_handoff nondim : {t_handoff:.12f}")
    print(f"t_handoff minutes: {nondim_time_to_minutes(t_handoff):.6f}")
    print(f"label            : {label} ({label_name(label)})")
    print(f"state_handoff    : {state_handoff}")

    plot_handoff_debug(
        cfg=cfg,
        env=env,
        post_tli_traj=post_tli_traj,
        post_tli_t_hist=post_tli_t_hist,
        mods=mods,
        out_dir=out_dir,
        policy_path=policy_path,
    )

    default_name = f"{policy_path.stem}_staged_handoff_{post_tli_minutes:g}min"
    user_name = ask_str("Name for PPO-B handoff library", default_name)
    stem = sanitize_name(user_name)

    default_output_dir = script_dir / "rough_scenario_classification"
    output_dir_str = ask_str("Output folder", str(default_output_dir))
    output_dir = ensure_dir(Path(output_dir_str).expanduser().resolve())

    npz_path = output_dir / f"{stem}.npz"

    if npz_path.exists():
        overwrite = ask_yes_no(f"File exists: {npz_path}\nOverwrite?", default=False)
        if not overwrite:
            npz_path = output_dir / f"{stem}_{timestamp_str()}.npz"
            print(f"Using new file:\n{npz_path}")

    npz_path = save_handoff_npz(
        out_path=npz_path,
        policy_path=policy_path,
        run_dir=run_dir,
        recovered=recovered,
        state_handoff=state_handoff,
        t_handoff=t_handoff,
        state_after_tli=state_after_tli,
        t_tli=t_tli,
        tli_time_source=tli_time_source,
        post_tli_minutes=post_tli_minutes,
        label=label,
        env=env,
        rollout_result=rollout_result,
        post_tli_traj=post_tli_traj,
        post_tli_t_hist=post_tli_t_hist,
    )

    json_path = save_metadata_json(
        npz_path=npz_path,
        policy_path=policy_path,
        run_dir=run_dir,
        recovered=recovered,
        post_tli_minutes=post_tli_minutes,
        label=label,
        t_tli=t_tli,
        t_handoff=t_handoff,
        tli_time_source=tli_time_source,
        env=env,
        rollout_result=rollout_result,
    )

    save_debug_json(
        out_dir=out_dir,
        policy_path=policy_path,
        run_dir=run_dir,
        recovered=recovered,
        env=env,
        rollout_result=rollout_result,
        state_after_tli=state_after_tli,
        state_handoff=state_handoff,
        t_tli=t_tli,
        t_handoff=t_handoff,
        post_tli_minutes=post_tli_minutes,
        npz_path=npz_path,
    )

    validate_saved_handoff_library(npz_path)

    print("\n" + "=" * 90)
    print("PPO-B handoff library saved")
    print("=" * 90)
    print(f"NPZ : {npz_path}")
    print(f"JSON: {json_path}")
    print(f"Debug/plots folder: {out_dir}")

    try:
        rel_npz = npz_path.relative_to(script_dir)
    except Exception:
        rel_npz = npz_path

    print("\nTo train PPO-B from this exact handoff, edit curriculum_ppob.py:")
    print("-" * 70)
    print(f'MAIN_LIB = "{rel_npz}"')
    print("MAIN_CASE_IDX = 0")
    print("")
    print("And keep these settings in the PPO-B stages:")
    print('ppo_b_case_source = "scenario_library"')
    print("ppo_b_use_fixed_index = True")
    print("ppo_b_fixed_index = MAIN_CASE_IDX")
    print("")
    print("Why index 0?")
    print("This script saves a one-row handoff library, so the only case is row 0.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
