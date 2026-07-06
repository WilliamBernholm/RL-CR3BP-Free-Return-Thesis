"""
============================================================
PPO TRAINING AND EVALUATION SCRIPT FOR EARTH-MOON FREE-RETURN
============================================================

This script is the main experiment runner for the planar Earth-Moon
CR3BP reinforcement learning project.

It coordinates the full workflow for:
- PPO-A training (TLI optimization)
- PPO-B training (MCC / correction optimization)
- periodic evaluation and diagnostics
- checkpointing and milestone saves
- plot generation and run summaries

------------------------------------------------------------
WHAT THIS SCRIPT DOES
------------------------------------------------------------

This script:
- selects the training profile and curriculum (PPO-A or PPO-B)
- applies profile-specific overrides from the curriculum files
- builds training and evaluation environments from cr3bp_env_v4.py
- creates or resumes a recurrent PPO model
- runs stage-by-stage curriculum training
- evaluates the model at fixed rollout intervals
- saves run configurations, checkpoints, and milestone models
- generates trajectory plots, evaluation summaries, and training diagnostics
- supports helper tools such as manual override mode, timelapse animation,
  checkpoint continuation, and batch evaluation of saved policies

------------------------------------------------------------
SUPPORTED TRAINING MODES
------------------------------------------------------------

PPO-A:
- trains the translunar injection problem
- starts from a spacecraft state in circular LEO around Earth
- the initial LEO position is defined by the spawn angle
- the policy learns the TLI burn behavior and timing

PPO-B:
- trains the mid-course correction problem
- starts from a known post-TLI handoff state
- the initial condition is loaded from a saved scenario library
- the policy learns MCC behavior that preserves or improves
  lunar flyby and Earth return performance

------------------------------------------------------------
CUSTOM RL BACKEND
------------------------------------------------------------

This script uses a custom recurrent PPO stack rather than a plain
baseline SB3 configuration.

Current backend components:
- TimeAwareRecurrentPPOv2
- TimeAwareRecurrentRolloutBuffer
- SquashedMlpLstmPolicy

(Changed from baseline: custom recurrent PPO implementation instead of
a standard off-the-shelf recurrent PPO training path.)

(Changed from baseline: time-aware rollout buffer with per-step dt_ratio
support and time-aware return / advantage computation.)

(Changed from baseline: tanh-squashed recurrent Gaussian policy for
bounded continuous actions.)

These backend components are intended to better match the variable-time
mission structure and bounded burn-action space used in this project.

------------------------------------------------------------
EVALUATION AND LOGGING
------------------------------------------------------------

During training, this script performs repeated evaluation runs and records:
- total reward statistics
- flyby and return performance
- ballistic proxy metrics
- delta-v usage
- success / failure rates
- milestone improvements across training
- trajectory plots and reward summaries
- PPO training metrics when available

(Changed from baseline: richer evaluation bookkeeping, milestone-based
checkpoint saving, and detailed run configuration export.)

(Changed from baseline: explicit separation between ballistic proxy
success, actual controlled trajectory success, and corridor / flyby
metrics.)

All outputs are saved to the configured run directory inside the
saved-policy root.

------------------------------------------------------------
WHAT THIS SCRIPT DOES NOT CONTAIN
------------------------------------------------------------

This script does not define:
- CR3BP dynamics
- propagation logic
- reward function internals
- reset / spawn physics
- plotting implementations
- low-level recurrent PPO buffer / policy internals

Those are provided by:
- cr3bp_env_v4.py
- cr3bp_plotting_v4.py
- config.py
- curriculum_ppoa.py
- curriculum_ppob.py
- time_aware_ppo_recurrent_V2.py
- time_aware_buffers_V2.py
- squashed_recurrent_policy.py

------------------------------------------------------------
DESIGN INTENT
------------------------------------------------------------

The purpose of this script is to keep the full experiment pipeline
in one place while keeping environment physics, plotting, and custom
RL backend internals separated.

This makes it easier to:
- train PPO-A and PPO-B from the same entry point
- compare curricula and reward settings
- resume old checkpoints
- reproduce runs from saved configuration files
- inspect learning progress with consistent diagnostics

============================================================
"""



from __future__ import annotations

import sys
import copy
import subprocess
import inspect
import math
import glob
import time
import re
import torch as th
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from config import (
    RUN,
    RunConfig,
    RewardConfig,
    RewardWeights,
    CurriculumStage,
    CR3BPConfig,
    ppo_rollout_block_size,
    apply_overrides,
)

from curriculum_ppoa import build_curriculum_ppoa
from curriculum_ppob import build_curriculum_ppob

from cr3bp_env_v4 import (
    RewardFunction,
    CR3BPFreeReturnEnv,
    apply_stage_to_cfg,
    build_reward_factory,
    kms_to_nondim_dv,
    minutes_to_nondim_time,
    nondim_time_to_minutes,
    global_burn_cap_nondim,
    tli_ballistic_trigger_nondim,
    cr3bp_vstar_kms,
    dist_to_interval,
    snap_curriculum_timesteps,
    get_obs_schema,
)

from cr3bp_plotting_v4 import (
    plot_trajectory,
    plot_trajectory_mcc_debug,
    plot_trajectory_earth_centered_inertial,
    collect_episode_reward_timeseries,
    save_episode_report_txt,
    plot_spawn_theta_sweep,
    save_spawn_theta_sweep_txt,
    plot_eval_trajectories_grid,
    plot_training_eval_reward_curve,
    plot_free_return_rates_curve,
    plot_ppo_metrics_curve,
    audit_reward_records,
    save_eval_episode_archive_npz_json,
    save_training_history_npz,
    plot_mean_eval_dv_curve,
)

# ============================================================
# PRESELECTED CHECKPOINT FOR UBUNTU / SERVER RUNS
# Change ONLY these two lines in the future if needed.
# ============================================================

SAVED_POLICY_BATCH_EVAL_EPISODES = 100

PRESELECTED_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent / "checkpoints" / "good_tli"
)

PRESELECTED_CHECKPOINT_FILE = (
    "MILESTONE_BEST_MEAN_BALLISTIC_123.969_STEP_22528__2026-04-06_17-51-20.zip"
)




def launch_manual_override_env():
    """
    Launch the standalone manual override environment in a new Python process.
    The manual tool uses the same CR3BP environment and action decoding as this trainer.
    """
    script_dir = Path(__file__).resolve().parent
    viewer_path = script_dir / "manual_override_env_3_3_2.py"

    if not viewer_path.exists():
        print(f"Manual override env script not found: {viewer_path}")
        return

    print(f"Launching manual override env: {viewer_path}")
    subprocess.run([sys.executable, str(viewer_path)], check=False)


def launch_timelapse_animator():
    """
    Launch the standalone timelapse animator script in a new Python process.
    Assumes CR3BP_Timelapse_V2_9.py is in the same folder as this trainer.
    """
    script_dir = Path(__file__).resolve().parent
    animator_path = script_dir / "CR3BP_Timelapse_V3_1.py"

    if not animator_path.exists():
        print(f"Animator script not found: {animator_path}")
        return

    print(f"Launching animator: {animator_path}")
    subprocess.run([sys.executable, str(animator_path)], check=False)


# ============================================================
# 6) TRAINING + EVALUATION
# ============================================================


def apply_stage_log_std_override(model, stage) -> None:
    """
    Optionally override the policy log-std using curriculum-stage settings.

    If stage.use_manual_log_std is False:
        do nothing

    If True:
        set all policy log_std entries to stage.manual_log_std_value
    """
    use_override = bool(getattr(stage, "use_manual_log_std", False))
    if not use_override:
        return

    log_std_value = float(getattr(stage, "manual_log_std_value", 0.0))

    if not hasattr(model, "policy") or not hasattr(model.policy, "log_std"):
        print("[WARN] Stage requested manual log_std override, but model.policy.log_std was not found.")
        return

    with th.no_grad():
        model.policy.log_std.fill_(log_std_value)
        std_mean = th.exp(model.policy.log_std).mean().item()

    print(
        f"[STAGE LOG_STD OVERRIDE] "
        f"stage={getattr(stage, 'name', 'unknown')} "
        f"use_manual_log_std={use_override} "
        f"log_std={log_std_value:.6f} "
        f"std_mean={std_mean:.6f}"
    )


def debug_raw_vs_clipped_action(model, obs, lstm_states, episode_start, deterministic=True):
    """
    Compare raw policy output from _predict() against the final action from predict().
    This reveals whether clipping is happening inside SB3/sb3-contrib before env.step().
    """
    policy = model.policy

    # Convert obs exactly like predict() does
    obs_tensor, vectorized_env = policy.obs_to_tensor(obs)

    if isinstance(obs_tensor, dict):
        n_envs = obs_tensor[next(iter(obs_tensor.keys()))].shape[0]
    else:
        n_envs = obs_tensor.shape[0]

    # Build recurrent state tensors exactly like policy.predict()
    if lstm_states is None:
        state = np.concatenate(
            [np.zeros(policy.lstm_hidden_state_shape) for _ in range(n_envs)],
            axis=1,
        )
        lstm_states_np = (state, state)
    else:
        lstm_states_np = lstm_states

    if episode_start is None:
        episode_start = np.array([False for _ in range(n_envs)], dtype=bool)

    with th.no_grad():
        lstm_states_th = (
            th.tensor(lstm_states_np[0], dtype=th.float32, device=policy.device),
            th.tensor(lstm_states_np[1], dtype=th.float32, device=policy.device),
        )
        episode_starts_th = th.tensor(episode_start, dtype=th.float32, device=policy.device)

        # RAW actions straight from _predict(), before predict() clips
        raw_actions_th, new_states_th = policy._predict(
            obs_tensor,
            lstm_states=lstm_states_th,
            episode_starts=episode_starts_th,
            deterministic=deterministic,
        )

    raw_actions = raw_actions_th.cpu().numpy().reshape((-1, *policy.action_space.shape))

    # FINAL actions returned by predict(), after internal clipping/unscaling
    final_actions, new_states_np = model.predict(
        obs,
        state=lstm_states,
        episode_start=episode_start,
        deterministic=deterministic,
    )

    print("\n========== ACTION SOURCE DEBUG ==========")
    print("Raw _predict() action      :", raw_actions)
    print("Raw min/max                :", raw_actions.min(), raw_actions.max())
    print("Final model.predict action :", final_actions)
    print("Final min/max              :", np.min(final_actions), np.max(final_actions))
    print("Would raw be clipped?      :", np.any(np.abs(raw_actions) > 1.0))
    print("Difference raw-final       :", raw_actions.squeeze() - np.array(final_actions).squeeze())
    print("=========================================\n")

    return final_actions, new_states_np

def attach_fresh_logger(model, run_dir: Path):
    from stable_baselines3.common.logger import configure

    tb_dir = ensure_dir(run_dir / "tb")
    new_logger = configure(str(tb_dir), ["stdout", "tensorboard"])
    model.set_logger(new_logger)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def policy_label_from_trainer_mode(trainer_mode: str) -> str:
    mode = str(trainer_mode).lower()
    if mode == "ppo_a":
        return "PPOA"
    if mode in ("ppo_b_baseline", "ppo_b_from_external_ic", "ppo_b_library"):
        return "PPOB"
    return "PPOX"


def stage_policy_label(stage: CurriculumStage) -> str:
    return policy_label_from_trainer_mode(getattr(stage, "trainer_mode", "ppo_a"))


def cfg_policy_label(cfg: CR3BPConfig) -> str:
    return policy_label_from_trainer_mode(getattr(cfg, "trainer_mode", "ppo_a"))


def get_saved_root(script_path: str) -> Path:
    return Path(script_path).resolve().parent / RUN.saved_root_name


def get_preselected_checkpoint_path() -> Path:
    """
    Return the hardcoded checkpoint path used for server-side
    non-interactive continuation runs.

    Change PRESELECTED_CHECKPOINT_DIR and PRESELECTED_CHECKPOINT_FILE
    at the top of this file to point to a different checkpoint later.
    """
    checkpoint_path = PRESELECTED_CHECKPOINT_DIR / PRESELECTED_CHECKPOINT_FILE

    print("\n" + "=" * 78)
    print("PRESELECTED CHECKPOINT MODE")
    print("=" * 78)
    print("To change the preselected save file in the future, edit these variables:")
    print("  PRESELECTED_CHECKPOINT_DIR")
    print("  PRESELECTED_CHECKPOINT_FILE")
    print("")
    print(f"Checkpoint directory : {PRESELECTED_CHECKPOINT_DIR}")
    print(f"Checkpoint filename  : {PRESELECTED_CHECKPOINT_FILE}")
    print(f"Resolved path        : {checkpoint_path}")
    print(f"Exists               : {checkpoint_path.exists()}")
    print("=" * 78 + "\n")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Preselected checkpoint not found.\n"
            f"Expected file:\n{checkpoint_path}\n\n"
            "Edit PRESELECTED_CHECKPOINT_DIR and PRESELECTED_CHECKPOINT_FILE "
            "at the top of train_ppo.py."
        )

    return checkpoint_path




def append_backend_info_to_run_config(run_config_path: Path, model) -> None:
    """
    Append the ACTUAL PPO/buffer backend info to the saved run_config.txt.
    """
    info = get_model_backend_info(model)

    with open(run_config_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=== actual_rl_backend ===\n")
        f.write(f"ppo_class_name = {info['ppo_class_name']}\n")
        f.write(f"ppo_module_name = {info['ppo_module_name']}\n")
        f.write(f"ppo_file = {info['ppo_file']}\n")
        f.write(f"buffer_class_name = {info['buffer_class_name']}\n")
        f.write(f"buffer_module_name = {info['buffer_module_name']}\n")
        f.write(f"buffer_file = {info['buffer_file']}\n")


def save_run_configuration_txt(
    run_dir: Path,
    cfg: CR3BPConfig,
    reward_cfg: RewardConfig,
    curriculum: list[CurriculumStage],
    resume_source: Optional[str] = None,
):
    lines = []

    lines.append("=" * 78)
    lines.append("RUN CONFIGURATION")
    lines.append("=" * 78)
    lines.append(f"timestamp: {timestamp_str()}")
    lines.append("")

    lines.append("[RUN]")
    lines.append(f"resume_source = {resume_source}")
    lines.append(f"enable_plotting = {RUN.enable_plotting}")
    lines.append(f"saved_root_name = {RUN.saved_root_name}")
    lines.append(f"total_timesteps = {RUN.total_timesteps}")
    lines.append(f"n_envs = {RUN.n_envs}")
    lines.append(f"train_seed = {RUN.train_seed}")
    lines.append(f"eval_seed = {RUN.eval_seed}")
    lines.append(f"eval_interval_steps = {RUN.eval_interval_steps}")
    lines.append(f"eval_episodes = {RUN.eval_episodes}")
    lines.append(f"plot_every_evals = {RUN.plot_every_evals}")
    lines.append(f"checkpoint_every = {RUN.checkpoint_every}")
    lines.append("")

    lines.append("[PPO-LSTM]")
    lines.append(f"gamma = {RUN.gamma}")
    lines.append(f"gae_lambda = {RUN.gae_lambda}")
    lines.append(f"n_steps = {RUN.n_steps}")
    lines.append(f"batch_size = {RUN.batch_size}")
    lines.append(f"n_epochs = {RUN.n_epochs}")
    lines.append(f"learning_rate = {RUN.learning_rate}")
    lines.append(f"clip_range = {RUN.clip_range}")
    lines.append(f"max_grad_norm = {RUN.max_grad_norm}")
    lines.append(f"ent_coef_default = {RUN.ent_coef_default}")
    lines.append(f"device = {RUN.device}")
    lines.append("")

    lines.append("[UNITS / CONVERSIONS]")
    lines.append(f"cr3bp_Lstar_km = {RUN.cr3bp_Lstar_km}")
    lines.append(f"cr3bp_Tstar_s = {RUN.cr3bp_Tstar_s}")
    lines.append(f"cr3bp_Vstar_kms = {cr3bp_vstar_kms()}")
    lines.append("")

    lines.append("[ACTION MODEL]")
    lines.append("action_space = [ax, ay, tau_raw]")
    lines.append("burn semantics = direct planar burn vector")
    lines.append("timing semantics = burn first, then drift")
    lines.append("")

    lines.append("[DRIFT MODEL]")
    lines.append(f"drift_min_minutes_pre_tli = {RUN.drift_min_minutes_pre_tli}")
    lines.append(f"drift_max_minutes_pre_tli = {RUN.drift_max_minutes_pre_tli}")
    lines.append(f"drift_min_nondim_pre_tli = {minutes_to_nondim_time(RUN.drift_min_minutes_pre_tli)}")
    lines.append(f"drift_max_nondim_pre_tli = {minutes_to_nondim_time(RUN.drift_max_minutes_pre_tli)}")
    lines.append(f"drift_min_minutes_post_tli = {RUN.drift_min_minutes_post_tli}")
    lines.append(f"drift_max_minutes_post_tli = {RUN.drift_max_minutes_post_tli}")
    lines.append(f"drift_min_nondim_post_tli = {minutes_to_nondim_time(RUN.drift_min_minutes_post_tli)}")
    lines.append(f"drift_max_nondim_post_tli = {minutes_to_nondim_time(RUN.drift_max_minutes_post_tli)}")
    lines.append(f"pre_tli_burn_deadzone_frac_of_tli_cap = {RUN.pre_tli_burn_deadzone_frac_of_tli_cap}")
    lines.append(f"no_tli_terminate_after_leo_orbits = {RUN.no_tli_terminate_after_leo_orbits}")
    lines.append("")

    lines.append("[BURN CAPS]")
    lines.append(f"use_global_burn_cap_kms = {RUN.use_global_burn_cap_kms}")
    lines.append(f"global_burn_cap_kms = {RUN.global_burn_cap_kms}")
    lines.append(f"global_burn_cap_nondim = {global_burn_cap_nondim()}")
    lines.append(f"use_single_dv_cap = {RUN.use_single_dv_cap}")
    lines.append(f"dv_cap_single = {RUN.dv_cap_single}")
    lines.append(f"tli_dv_max_kms = {RUN.tli_dv_max_kms}")
    lines.append(f"mcc_dv_max_kms = {RUN.mcc_dv_max_kms}")
    lines.append("")

    lines.append("[TLI BALLISTIC TRIGGER]")
    lines.append(f"tli_ballistic_scale = {RUN.tli_ballistic_scale}")
    lines.append(f"tli_ballistic_trigger_kms = {RUN.tli_ballistic_trigger_kms}")
    lines.append(f"tli_ballistic_trigger_nondim = {tli_ballistic_trigger_nondim()}")
    lines.append(f"tli_departure_trigger_rE = {RUN.tli_departure_trigger_rE}")
    lines.append("tli_commit_rule = commit when dv_mag >= trigger OR rE >= departure trigger")
    lines.append("")

    lines.append("[PROPAGATION]")
    lines.append(f"fine_substep_region_radius = {RUN.fine_substep_region_radius}")
    lines.append(f"fine_rk4_substep_minutes = {RUN.fine_rk4_substep_minutes}")
    lines.append(f"rk4_substep_target_min_minutes = {RUN.rk4_substep_target_min_minutes}")
    lines.append(f"rk4_substep_target_max_minutes = {RUN.rk4_substep_target_max_minutes}")
    lines.append(f"rk4_target_transition_min_minutes = {RUN.rk4_target_transition_min_minutes}")
    lines.append(f"rk4_target_transition_max_minutes = {RUN.rk4_target_transition_max_minutes}")
    lines.append("")

    lines.append("[CR3BP CONFIG BASE]")
    lines.append(f"mu = {cfg.mu}")
    lines.append(f"dt = {cfg.dt}")
    lines.append(f"t_max = {cfg.t_max}")
    lines.append(f"integration_substeps = {cfg.integration_substeps}")
    lines.append(f"r0_earth = {cfg.r0_earth}")
    lines.append(f"v_circ_earth = {cfg.v_circ_earth}")
    lines.append(f"r_moon_flyby = {cfg.r_moon_flyby}")
    lines.append(f"r_earth_return = {cfg.r_earth_return}")
    lines.append(f"r_earth_impact = {cfg.r_earth_impact}")
    lines.append(f"r_moon_impact = {cfg.r_moon_impact}")
    lines.append(f"r_escape = {cfg.r_escape}")
    lines.append(f"rp_min = {cfg.rp_min}")
    lines.append(f"rp_max = {cfg.rp_max}")
    lines.append(f"dv_max_tli = {cfg.dv_max_tli}")
    lines.append(f"dv_max_mcc = {cfg.dv_max_mcc}")
    lines.append(f"dv_noise_sigma_tli = {cfg.dv_noise_sigma_tli}")
    lines.append(f"dv_noise_sigma_mcc = {cfg.dv_noise_sigma_mcc}")
    lines.append(f"mcc_enabled = {cfg.mcc_enabled}")
    lines.append(f"store_dense_training_traj = {cfg.store_dense_training_traj}")
    lines.append(f"add_mode_obs = {cfg.add_mode_obs}")
    lines.append(f"add_legacy_mode_obs = {cfg.add_legacy_mode_obs}")
    lines.append(
        "legacy_mode_obs_fields = [tli_used_flag, tau_max_current_norm, dv_cap_current_norm, pre_tli_clock_norm]"
    )
    lines.append(f"pos_scale = {cfg.pos_scale}")
    lines.append(f"vel_scale = {cfg.vel_scale}")
    lines.append(f"c_scale = {cfg.c_scale}")
    lines.append(f"add_phase_angle_obs = {cfg.add_phase_angle_obs}")
    lines.append(f"trainer_mode = {cfg.trainer_mode}")
    lines.append(f"tli_control_mode = {cfg.tli_control_mode}")
    lines.append(f"ppo_b_baseline_theta = {cfg.ppo_b_baseline_theta}")
    lines.append(f"ppo_b_baseline_ax = {cfg.ppo_b_baseline_ax}")
    lines.append(f"ppo_b_baseline_ay = {cfg.ppo_b_baseline_ay}")
    lines.append(f"ppo_b_baseline_tau = {cfg.ppo_b_baseline_tau}")
    lines.append(f"ppo_b_baseline_state_noise_pos = {cfg.ppo_b_baseline_state_noise_pos}")
    lines.append(f"ppo_b_baseline_state_noise_vel = {cfg.ppo_b_baseline_state_noise_vel}")
    lines.append(f"ppo_b_case_source = {cfg.ppo_b_case_source}")
    lines.append(f"ppo_b_library_path = {cfg.ppo_b_library_path}")
    lines.append(f"ppo_b_prob_good = {cfg.ppo_b_prob_good}")
    lines.append(f"ppo_b_prob_savable = {cfg.ppo_b_prob_savable}")
    lines.append(f"ppo_b_prob_bad = {cfg.ppo_b_prob_bad}")
    lines.append(f"ppo_b_eval_use_same_distribution = {cfg.ppo_b_eval_use_same_distribution}")
    lines.append(f"ppo_b_noise_theta_deg = {cfg.ppo_b_noise_theta_deg}")
    lines.append(f"ppo_b_noise_tli_dir_deg = {cfg.ppo_b_noise_tli_dir_deg}")
    lines.append(f"ppo_b_noise_tli_dv_kms = {cfg.ppo_b_noise_tli_dv_kms}")
    lines.append(f"ppo_b_use_fixed_index = {cfg.ppo_b_use_fixed_index}")
    lines.append(f"ppo_b_fixed_index = {cfg.ppo_b_fixed_index}")
    lines.append(f"ppo_b_fixed_state_noise_pos = {cfg.ppo_b_fixed_state_noise_pos}")
    lines.append(f"ppo_b_fixed_state_noise_vel = {cfg.ppo_b_fixed_state_noise_vel}")
    lines.append(f"tli_only_mode = {cfg.tli_only_mode}")
    lines.append(f"reward_after_tli_ballistic_enabled = {cfg.reward_after_tli_ballistic_enabled}")
    lines.append(f"spawn_theta_limit_enabled = {cfg.spawn_theta_limit_enabled}")
    lines.append(f"spawn_theta_min = {cfg.spawn_theta_min}")
    lines.append(f"spawn_theta_max = {cfg.spawn_theta_max}")
    lines.append(f"terminate_on_dv_budget_exceed = {cfg.terminate_on_dv_budget_exceed}")
    lines.append("")

    lines.append("[REWARD CONFIG DEFAULTS]")
    rcfg = reward_cfg
    lines.append(f"v_target_moon = {rcfg.v_target_moon}")
    lines.append(f"v_deadzone = {rcfg.v_deadzone}")
    lines.append(f"beta_distance_flyby = {rcfg.beta_distance_flyby}")
    lines.append(f"r0_distance_flyby = {rcfg.r0_distance_flyby}")
    lines.append(f"beta_distance_return = {rcfg.beta_distance_return}")
    lines.append(f"r0_distance_return = {rcfg.r0_distance_return}")
    lines.append(f"dv_budget = {rcfg.dv_budget}")
    lines.append(f"dv_scale = {rcfg.dv_scale}")
    lines.append(f"earth_radius = {rcfg.earth_radius}")
    lines.append(f"moon_radius = {rcfg.moon_radius}")
    lines.append(f"flyby_reward_gate = {rcfg.flyby_reward_gate}")
    lines.append("")

    lines.append("[CURRICULUM]")
    for i, stage in enumerate(curriculum, start=1):
        w = stage.reward_weights
        lines.append(f"Stage {i}: {stage.name}")
        lines.append(f"  timesteps = {stage.timesteps}")
        lines.append(f"  trainer_mode = {stage.trainer_mode}")
        lines.append(f"  tli_control_mode = {stage.tli_control_mode}")
        lines.append(f"  entropy_coef = {stage.entropy_coef}")
        lines.append(f"  mcc_enabled = {stage.mcc_enabled}")
        lines.append(f"  tli_only_mode = {stage.tli_only_mode}")
        lines.append(f"  reward_after_tli_ballistic_enabled = {stage.reward_after_tli_ballistic_enabled}")
        lines.append(f"  spawn_theta_limit_enabled = {stage.spawn_theta_limit_enabled}")
        lines.append(f"  spawn_theta_min = {stage.spawn_theta_min}")
        lines.append(f"  spawn_theta_max = {stage.spawn_theta_max}")
        lines.append(f"  ppo_b_baseline_theta = {stage.ppo_b_baseline_theta}")
        lines.append(f"  ppo_b_baseline_ax = {stage.ppo_b_baseline_ax}")
        lines.append(f"  ppo_b_baseline_ay = {stage.ppo_b_baseline_ay}")
        lines.append(f"  ppo_b_baseline_tau = {stage.ppo_b_baseline_tau}")
        lines.append(f"  ppo_b_baseline_state_noise_pos = {stage.ppo_b_baseline_state_noise_pos}")
        lines.append(f"  ppo_b_baseline_state_noise_vel = {stage.ppo_b_baseline_state_noise_vel}")
        lines.append(f"  ppo_b_case_source = {stage.ppo_b_case_source}")
        lines.append(f"  ppo_b_library_path = {stage.ppo_b_library_path}")
        lines.append(f"  ppo_b_prob_good = {stage.ppo_b_prob_good}")
        lines.append(f"  ppo_b_prob_savable = {stage.ppo_b_prob_savable}")
        lines.append(f"  ppo_b_prob_bad = {stage.ppo_b_prob_bad}")
        lines.append(f"  ppo_b_eval_use_same_distribution = {stage.ppo_b_eval_use_same_distribution}")
        lines.append(f"  ppo_b_noise_theta_deg = {stage.ppo_b_noise_theta_deg}")
        lines.append(f"  ppo_b_noise_tli_dir_deg = {stage.ppo_b_noise_tli_dir_deg}")
        lines.append(f"  ppo_b_noise_tli_dv_kms = {stage.ppo_b_noise_tli_dv_kms}")
        lines.append(f"  ppo_b_use_fixed_index = {stage.ppo_b_use_fixed_index}")
        lines.append(f"  ppo_b_fixed_index = {stage.ppo_b_fixed_index}")
        lines.append(f"  ppo_b_fixed_state_noise_pos = {stage.ppo_b_fixed_state_noise_pos}")
        lines.append(f"  ppo_b_fixed_state_noise_vel = {stage.ppo_b_fixed_state_noise_vel}")
        lines.append(f"  dv_noise_sigma_tli = {stage.dv_noise_sigma_tli}")
        lines.append(f"  dv_noise_sigma_mcc = {stage.dv_noise_sigma_mcc}")
        lines.append(f"  use_manual_log_std = {stage.use_manual_log_std}")
        lines.append(f"  manual_log_std_value = {stage.manual_log_std_value}")
        lines.append(f"  w_flyby = {w.w_flyby}")
        lines.append(f"  w_velocity = {w.w_velocity}")
        lines.append(f"  w_dv = {w.w_dv}")
        lines.append(f"  w_return = {w.w_return}")
        lines.append(f"  w_budget = {w.w_budget}")
        lines.append(f"  w_escape = {w.w_escape}")
        lines.append(f"  w_earth_crash = {w.w_earth_crash}")
        lines.append(f"  w_moon_crash = {w.w_moon_crash}")
        lines.append(f"  w_postflyby_earth_crash = {w.w_postflyby_earth_crash}")
        lines.append(f"  w_invalid_preflyby_earth_return = {w.w_invalid_preflyby_earth_return}")
        lines.append("")

    out_path = run_dir / "run_config.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


def make_new_run_dir(script_path: str, run_label: str = "RUN") -> Path:
    root = get_saved_root(script_path)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{run_label}_{timestamp_str()}_run"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_eval_model_with_stats(
    model,
    run_dir: Path,
    stage_idx: int,
    step_count: int,
    reward_mean: float,
    success_rate: float,
    lunar_distance: float,
    corridor_miss: float,
    policy_label: Optional[str] = None,
) -> Path:
    if policy_label is None:
        policy_label = "Model"

    tag = (
        f"stage{int(stage_idx)+1:02d}"
        f"_step{int(step_count):08d}"
        f"_R{fmt_num(reward_mean,2)}"
        f"_SR{fmt_num(success_rate,3)}"
        f"_LD{fmt_num(lunar_distance,5)}"
        f"_CM{fmt_num(corridor_miss,5)}"
    )

    return save_model_timestamped(
        model,
        run_dir,
        tag=tag,
        policy_label=policy_label,
    )


def save_model_timestamped(
    model,
    run_dir: Path,
    tag: str,
    policy_label: Optional[str] = None,
) -> Path:
    if policy_label is None:
        policy_label = "POLICY"

    fname = f"{policy_label}__{tag}__{timestamp_str()}.zip"
    out_path = run_dir / fname
    model.save(str(out_path))
    return out_path


def format_seconds_to_hms(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"





def round_up_to_rollout_multiple(x: int) -> int:
    block = ppo_rollout_block_size()
    return int(math.ceil(float(x) / float(block)) * block)


def steps_until_next_multiple(current_steps: int, every_steps: int) -> int:
    if every_steps <= 0:
        return 0
    rem = current_steps % every_steps
    if rem == 0:
        return 0
    return int(every_steps - rem)


def list_policy_files(saved_root: Path) -> list[Path]:
    if not saved_root.exists():
        return []
    files = [Path(p) for p in glob.glob(str(saved_root / "*" / "*.zip"))]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def choose_from_list(paths: list[Path], title: str) -> Path:
    if len(paths) == 0:
        raise FileNotFoundError("No saved policy .zip files found.")
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for i, p in enumerate(paths):
        run_name = p.parent.name
        print(f"[{i:>3d}] {run_name} / {p.name}")
    while True:
        s = input("\nSelect index: ").strip()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(paths):
                return paths[idx]
        print("Invalid selection. Try again.")



def safe_mean(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def safe_mean(x):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def safe_std(x):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0
    return float(np.std(arr))


def safe_min(x):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.min(arr))


def safe_max(x):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.max(arr))


def fmt_num(x: float, prec: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{float(x):.{prec}f}"


def fmt_km(nd: float) -> str:
    if not np.isfinite(nd):
        return "nan"
    return f"{float(nd) * 384400.0:.1f}"


def run_eval_episode_collect(
    model,
    env,
    deterministic: bool = True,
    action_debug: bool = False,
    capture_plot_data: bool = True,
):
    """
    Roll one deterministic eval episode and collect:
    - exact reset fingerprint
    - exact first action
    - exact per-step reward records
    """
    obs, info = env.reset()
    done = False
    trunc = False

    # --------------------------------------------------------
    # RESET FINGERPRINT
    # --------------------------------------------------------
    reset_state = np.asarray(getattr(env, "state", []), dtype=np.float64).copy()
    reset_obs = np.asarray(obs, dtype=np.float64).copy()

    reset_debug = {
        "scenario_index": int(info.get("ppo_b_scenario_index", -1)),
        "scenario_row_index": int(info.get("ppo_b_scenario_row_index", -1)),
        "scenario_label": int(info.get("ppo_b_scenario_label", -1)),
        "scenario_term_reason": str(info.get("ppo_b_scenario_term_reason", "")),
        "scenario_fixed_index_mode": bool(info.get("ppo_b_scenario_fixed_index_mode", False)),
        "scenario_noise_pos_sigma": float(info.get("ppo_b_scenario_noise_pos_sigma", 0.0)),
        "scenario_noise_vel_sigma": float(info.get("ppo_b_scenario_noise_vel_sigma", 0.0)),
        "reset_state": reset_state,
        "reset_obs": reset_obs,
    }

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    ep_reward_sum = 0.0
    reward_terms_total: Dict[str, float] = {}

    reward_list: List[float] = []
    reward_records: List[Dict[str, Any]] = []

    min_rM_roll = np.inf
    vrel_at_min = np.nan

    first_action = None
    first_action_step = -1

    step_idx = 0
    while not (done or trunc):
        if action_debug:
            action, lstm_states = debug_raw_vs_clipped_action(
                model,
                obs,
                lstm_states,
                episode_start,
                deterministic=deterministic,
            )
        else:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )

        action_arr = np.asarray(action, dtype=np.float64).copy()
        if first_action is None:
            first_action = action_arr.copy()
            first_action_step = int(step_idx)

        obs, r, done, trunc, info = env.step(action)
        episode_start = np.array([done or trunc], dtype=bool)

        r = float(r)
        ep_reward_sum += r
        reward_list.append(r)

        record = info.get("reward_record", None)
        if not isinstance(record, dict):
            raise RuntimeError(
                "Missing info['reward_record'] during eval. "
                "Reward reporting is not trustworthy. "
                "Check _compute_reward_sean() and env.step() info propagation."
            )

        record_copy = copy.deepcopy(record)
        reward_records.append(record_copy)

        rec_terms = record_copy.get("terms", {})
        if not isinstance(rec_terms, dict):
            raise RuntimeError("reward_record['terms'] is missing or invalid.")

        for k, v in rec_terms.items():
            if isinstance(v, (int, float, np.floating, bool)):
                fv = float(v)
                if np.isfinite(fv):
                    reward_terms_total[k] = reward_terms_total.get(k, 0.0) + fv

        rM = float(info.get("rM", np.inf))
        vrel = float(info.get("vrel_moon", np.nan))
        if np.isfinite(rM) and rM < min_rM_roll:
            min_rM_roll = rM
            vrel_at_min = vrel

        step_idx += 1

    reason = str(info.get("term_reason", ""))

    flyby_done = bool(info.get("flyby_done", False))
    corridor_hit = bool(info.get("return_corridor_hit_postflyby", False))

    ballistic_success = bool(info.get("ballistic_tli_corridor_hit", False))
    trajectory_success = (reason == "success")
    success_flag_latched = bool(info.get("success", False))

    min_rM_i = float(info.get("min_rM", min_rM_roll))
    min_rE_i = float(info.get("min_rE", np.nan))
    min_rE_postflyby_i = float(info.get("min_rE_postflyby", np.nan))
    return_corridor_miss_i = float(info.get("best_postflyby_corridor_dist", np.nan))

    moon_corridor_miss_i = dist_to_interval(
        float(min_rM_i),
        float(env.cfg.r_moon_impact),
        float(env.cfg.r_moon_flyby),
    )

    all_term_keys = sorted({
        k
        for rec in reward_records
        for k in rec.get("terms", {}).keys()
    })

    terms_ts: Dict[str, np.ndarray] = {}
    T = len(reward_list)
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
        for i, rec in enumerate(reward_records):
            val = rec.get("metrics", {}).get(k, np.nan)
            try:
                fv = float(val)
            except Exception:
                fv = np.nan
            arr[i] = fv if np.isfinite(fv) else np.nan
        terms_ts[f"metric_{k}"] = arr

    audit = audit_reward_records(
        rewards=np.asarray(reward_list, dtype=np.float64),
        reward_records=reward_records,
    )

        
    if capture_plot_data:
        traj = np.asarray(getattr(env, "traj", []), dtype=np.float64)
        t_hist = np.asarray(getattr(env, "t_hist", []), dtype=np.float64)

        ballistic_ref_traj_raw = getattr(env, "ballistic_ref_traj", None)
        ballistic_ref_t_hist_raw = getattr(env, "ballistic_ref_t_hist", None)

        ballistic_ref_traj = (
            np.asarray(ballistic_ref_traj_raw, dtype=np.float64)
            if ballistic_ref_traj_raw is not None
            else None
        )
        ballistic_ref_t_hist = (
            np.asarray(ballistic_ref_t_hist_raw, dtype=np.float64)
            if ballistic_ref_t_hist_raw is not None
            else None
        )

        env_snapshot = copy.deepcopy(env)
    else:
        traj = np.zeros((0, 4), dtype=np.float64)
        t_hist = np.zeros((0,), dtype=np.float64)
        ballistic_ref_traj = None
        ballistic_ref_t_hist = None
        env_snapshot = None

    
    action_history = copy.deepcopy(getattr(env, "action_history", []))
    obs_schema = get_obs_schema(env)

    terminal_marker_rot = (
        np.asarray(traj[-1, :2], dtype=np.float64).copy()
        if traj.ndim == 2 and traj.shape[0] > 0 and traj.shape[1] >= 2
        else np.zeros((0,), dtype=np.float64)
    )

    ballistic_terminal_marker_rot = (
        np.asarray(ballistic_ref_traj[-1, :2], dtype=np.float64).copy()
        if ballistic_ref_traj is not None
        and ballistic_ref_traj.ndim == 2
        and ballistic_ref_traj.shape[0] > 0
        and ballistic_ref_traj.shape[1] >= 2
        else np.zeros((0,), dtype=np.float64)
    )

    return {
        "reason": reason,
        "reward_sum": float(ep_reward_sum),
        "rewards": np.asarray(reward_list, dtype=np.float64),
        "reward_records": reward_records,
        "reward_audit": audit,
        "terms_ts": terms_ts,
        "reward_terms_total": reward_terms_total,
        "flyby_done": bool(flyby_done),
        "corridor_hit": bool(corridor_hit),
        "ballistic_success": bool(ballistic_success),
        "trajectory_success": bool(trajectory_success),
        "success_strict": bool(trajectory_success),
        "success_flag_latched": bool(success_flag_latched),
        "left_leo": bool(info.get("left_leo", False)),
        "dv_used": float(info.get("dv_used", np.nan)),
        "dv0": float(info.get("dv0", np.nan)),
        "min_rM": float(min_rM_i),
        "min_rE": float(min_rE_i),
        "min_rE_postflyby": float(min_rE_postflyby_i),
        "moon_corridor_miss": float(moon_corridor_miss_i),
        "return_corridor_miss": float(return_corridor_miss_i),
        "vrel_at_min_rM": float(info.get("vrel_at_min_rM", vrel_at_min)),
        "traj": traj,
        "t_hist": t_hist,
        "ballistic_ref_traj": ballistic_ref_traj,
        "ballistic_ref_t_hist": ballistic_ref_t_hist,
        "env_snapshot": env_snapshot,
        "info_last": info,
        "left_leo_step": float(info.get("left_leo_step", np.nan)),
        "tli_tau": float(info.get("tli_tau", np.nan)),
        "tli_ax": float(info.get("tli_ax", np.nan)),
        "tli_ay": float(info.get("tli_ay", np.nan)),
        "ballistic_tli_reward": float(info.get("ballistic_tli_reward", np.nan)),
        "ballistic_tli_min_rM": float(info.get("ballistic_tli_min_rM", np.nan)),
        "ballistic_tli_corridor_dist": float(info.get("ballistic_tli_corridor_dist", np.nan)),
        "ballistic_tli_corridor_hit": bool(info.get("ballistic_tli_corridor_hit", False)),
        "burns": np.asarray(getattr(env, "burns", []), dtype=np.float64),
        "burn_events": copy.deepcopy(getattr(env, "burn_events", [])),
        "mcc_ballistic_overlays": copy.deepcopy(getattr(env, "mcc_ballistic_overlays", [])),
        "action_history": action_history,
        "obs_schema": list(obs_schema),
        "obs_dim": int(len(obs_schema)),
        "state_schema": ["x", "y", "vx", "vy"],
        "state_dim": 4,
        "action_schema": ["ax_raw", "ay_raw", "tau_raw"],
        "terminal_marker_rot": terminal_marker_rot,
        "ballistic_terminal_marker_rot": ballistic_terminal_marker_rot,

        # ---------------- debug ----------------
        "reset_debug": reset_debug,
        "first_action": np.asarray(first_action, dtype=np.float64) if first_action is not None else np.zeros((0,), dtype=np.float64),
        "first_action_step": int(first_action_step),
    }


def aggregate_reward_term_stats(eval_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Build mean/std/min/max for episode-total reward terms across the eval batch.
    """
    all_keys = set()
    for ep in eval_results:
        all_keys.update(ep["reward_terms_total"].keys())

    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(all_keys):
        vals = []
        for ep in eval_results:
            vals.append(float(ep["reward_terms_total"].get(k, 0.0)))
        out[k] = {
            "mean": safe_mean(vals),
            "std": safe_std(vals),
            "min": safe_min(vals),
            "max": safe_max(vals),
        }
    return out

def _fmt_eval_num(x: float, prec: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{float(x):.{prec}f}"


def _fmt_eval_km(x_nd: float, em_distance_km: float = 384400.0) -> str:
    if not np.isfinite(x_nd):
        return "nan"
    return f"{float(x_nd) * float(em_distance_km):.1f}"


def build_eval_summary_text(
    *,
    eval_idx: int,
    step_count: int,
    n_episodes: int,
    reasons: Dict[str, int],

    sr: float,
    trajectory_success_rate: float,
    flyby_rate: float,
    corridor_hit_rate: float,
    success_flag_rate: float,
    left_leo_rate: float,
    left_leo_step_mean: float,
    

    reward_mean: float,
    reward_std: float,
    best_reward_ever: float,
    best_reward_eval: int,
    best_reward_step: int,

    dv_mean: float,
    dv_std: float,
    dv0_mean: float,
    dv0_std: float,

    minrM_mean: float,
    minrM_std: float,
    vrel_min_mean: float,
    vrel_min_std: float,
    best_min_rM_ever: float,
    best_min_rM_eval: int,
    best_min_rM_step: int,
    best_moon_corridor_miss_ever: float,
    best_moon_corridor_miss_eval: int,
    best_moon_corridor_miss_step: int,

    minrE_mean: float,
    minrE_std: float,
    minrE_postflyby_mean: float,
    minrE_postflyby_std: float,
    return_corridor_miss_mean: float,
    return_corridor_miss_std: float,
    best_return_corridor_miss_ever: float,
    best_return_corridor_miss_eval: int,
    best_return_corridor_miss_step: int,
    best_postflyby_rE_ever: float,
    best_postflyby_rE_eval: int,
    best_postflyby_rE_step: int,

    best_single_moon_corridor_miss_ever: float,
    best_single_moon_corridor_miss_eval: int,
    best_single_moon_corridor_miss_step: int,

    best_single_return_corridor_miss_ever: float,
    best_single_return_corridor_miss_eval: int,
    best_single_return_corridor_miss_step: int,

    ballistic_tli_reward_mean: float,
    ballistic_tli_reward_std: float,
    best_ballistic_tli_reward_ever: float,
    best_ballistic_tli_reward_eval: int,
    best_ballistic_tli_reward_step: int,
    best_single_ballistic_tli_reward_ever: float,
    best_single_ballistic_tli_reward_eval: int,
    best_single_ballistic_tli_reward_step: int,

    moon_corridor_miss_mean: float,
    moon_corridor_miss_std: float,

    ballistic_tli_min_rM_mean: float,
    ballistic_tli_min_rM_std: float,
    ballistic_tli_corridor_dist_mean: float,
    ballistic_tli_corridor_dist_std: float,
    ballistic_tli_corridor_hit_rate: float,
    best_ballistic_tli_min_rM_ever: float,
    best_ballistic_tli_min_rM_eval: int,
    best_ballistic_tli_min_rM_step: int,
    best_ballistic_tli_corridor_dist_ever: float,
    best_ballistic_tli_corridor_dist_eval: int,
    best_ballistic_tli_corridor_dist_step: int,
    best_ballistic_tli_corridor_hit_rate_ever: float,

    best_success_rate_ever: float,
    best_flyby_rate_ever: float,
    best_corridor_hit_rate_ever: float,


    best_trajectory_success_rate_ever: float,
    best_success_flag_rate_ever: float,

    preservation_rate: float,
    degradation_rate: float,
    rescue_rate: float,
    unchanged_bad_rate: float,

    best_preservation_rate_ever: float,
    best_degradation_rate_ever: float,
    best_rescue_rate_ever: float,
    best_unchanged_bad_rate_ever: float,
    

    total_eval_episodes_seen: int,
    total_flyby_episodes: int,
    total_corridor_hit_episodes: int,
    total_success_episodes: int,
    first_flyby_eval: int,
    first_flyby_step: int,
    first_corridor_hit_eval: int,
    first_corridor_hit_step: int,
    first_success_eval: int,
    first_success_step: int,

    tli_tau_mean: float,
    tli_tau_std: float,
    tli_ax_mean: float,
    tli_ax_std: float,
    tli_ay_mean: float,
    tli_ay_std: float,

    reward_term_stats: Dict[str, Dict[str, float]],
    em_distance_km: float = 384400.0,
) -> str:
    lines = []

    lines.append("=" * 108)
    lines.append(f"EVAL #{eval_idx} | step={step_count:,} | episodes={n_episodes}")
    lines.append("=" * 108)

    lines.append("RATES")
    lines.append("-" * 108)
    lines.append(f"success_rate                 : {_fmt_eval_num(sr, 3)}   | best ever: {_fmt_eval_num(best_success_rate_ever, 3)}   (ballistic IC success)")
    lines.append(f"trajectory_success_rate      : {_fmt_eval_num(trajectory_success_rate, 3)}   (actual controlled trajectory)")
    lines.append(f"flyby_rate                   : {_fmt_eval_num(flyby_rate, 3)}   | best ever: {_fmt_eval_num(best_flyby_rate_ever, 3)}")
    lines.append(f"corridor_hit_rate            : {_fmt_eval_num(corridor_hit_rate, 3)}   | best ever: {_fmt_eval_num(best_corridor_hit_rate_ever, 3)}")
    lines.append(f"success_flag_rate            : {_fmt_eval_num(success_flag_rate, 3)}")
    lines.append(f"preservation_rate            : {_fmt_eval_num(preservation_rate, 3)}   | best ever: {_fmt_eval_num(best_preservation_rate_ever, 3)}")
    lines.append(f"degradation_rate             : {_fmt_eval_num(degradation_rate, 3)}   | best ever: {_fmt_eval_num(best_degradation_rate_ever, 3)}")
    lines.append(f"rescue_rate                  : {_fmt_eval_num(rescue_rate, 3)}   | best ever: {_fmt_eval_num(best_rescue_rate_ever, 3)}")
    lines.append(f"unchanged_bad_rate           : {_fmt_eval_num(unchanged_bad_rate, 3)}   | best ever: {_fmt_eval_num(best_unchanged_bad_rate_ever, 3)}")
    lines.append(f"left_leo_rate                : {_fmt_eval_num(left_leo_rate, 3)}")
    lines.append(f"termination_reasons          : {reasons}")

    lines.append("")
    lines.append("REWARDS")
    lines.append("-" * 108)
    lines.append(f"episode_total_reward         : mean={_fmt_eval_num(reward_mean,4)}  std={_fmt_eval_num(reward_std,4)}")
    lines.append(f"best_single_reward_ever      : {_fmt_eval_num(best_reward_ever,4)}   (eval {best_reward_eval}, step {best_reward_step:,})")
    lines.append(f"ballistic_reward             : mean={_fmt_eval_num(ballistic_tli_reward_mean,4)}  std={_fmt_eval_num(ballistic_tli_reward_std,4)}")
    lines.append(f"best_mean_ballistic_ever     : {_fmt_eval_num(best_ballistic_tli_reward_ever,4)}   (eval {best_ballistic_tli_reward_eval}, step {best_ballistic_tli_reward_step:,})")
    lines.append(f"best_single_ballistic_ever   : {_fmt_eval_num(best_single_ballistic_tli_reward_ever,4)}   (eval {best_single_ballistic_tli_reward_eval}, step {best_single_ballistic_tli_reward_step:,})")

    lines.append("")
    lines.append("MOON GEOMETRY")
    lines.append("-" * 108)
    lines.append(f"moon_corridor_miss           : mean={_fmt_eval_num(moon_corridor_miss_mean,6)}  std={_fmt_eval_num(moon_corridor_miss_std,6)}   [{_fmt_eval_km(moon_corridor_miss_mean, em_distance_km)} km]")
    lines.append(f"min_rM                       : mean={_fmt_eval_num(minrM_mean,6)}  std={_fmt_eval_num(minrM_std,6)}   [{_fmt_eval_km(minrM_mean, em_distance_km)} km]")
    lines.append(f"vrel_at_min_rM               : mean={_fmt_eval_num(vrel_min_mean,4)}  std={_fmt_eval_num(vrel_min_std,4)}")
    lines.append(f"best_min_rM_ever             : {_fmt_eval_num(best_min_rM_ever,6)}   [{_fmt_eval_km(best_min_rM_ever, em_distance_km)} km]   (eval {best_min_rM_eval}, step {best_min_rM_step:,})")
    lines.append(f"best_moon_miss_ever          : {_fmt_eval_num(best_moon_corridor_miss_ever,6)}   [{_fmt_eval_km(best_moon_corridor_miss_ever, em_distance_km)} km]   (eval {best_moon_corridor_miss_eval}, step {best_moon_corridor_miss_step:,})")
    lines.append(f"best_single_moon_miss_ever   : {_fmt_eval_num(best_single_moon_corridor_miss_ever,6)}   [{_fmt_eval_km(best_single_moon_corridor_miss_ever, em_distance_km)} km]   (eval {best_single_moon_corridor_miss_eval}, step {best_single_moon_corridor_miss_step:,})")

    lines.append("")
    lines.append("EARTH RETURN GEOMETRY")
    lines.append("-" * 108)
    lines.append(f"min_rE                       : mean={_fmt_eval_num(minrE_mean,6)}  std={_fmt_eval_num(minrE_std,6)}   [{_fmt_eval_km(minrE_mean, em_distance_km)} km]")
    lines.append(f"min_rE_postflyby             : mean={_fmt_eval_num(minrE_postflyby_mean,6)}  std={_fmt_eval_num(minrE_postflyby_std,6)}   [{_fmt_eval_km(minrE_postflyby_mean, em_distance_km)} km]")
    lines.append(f"return_corridor_miss         : mean={_fmt_eval_num(return_corridor_miss_mean,6)}  std={_fmt_eval_num(return_corridor_miss_std,6)}   [{_fmt_eval_km(return_corridor_miss_mean, em_distance_km)} km]")
    lines.append(f"best_return_miss_ever        : {_fmt_eval_num(best_return_corridor_miss_ever,6)}   [{_fmt_eval_km(best_return_corridor_miss_ever, em_distance_km)} km]   (eval {best_return_corridor_miss_eval}, step {best_return_corridor_miss_step:,})")
    lines.append(f"best_postflyby_rE_ever       : {_fmt_eval_num(best_postflyby_rE_ever,6)}   [{_fmt_eval_km(best_postflyby_rE_ever, em_distance_km)} km]   (eval {best_postflyby_rE_eval}, step {best_postflyby_rE_step:,})")
    lines.append(f"best_single_return_miss_ever : {_fmt_eval_num(best_single_return_corridor_miss_ever,6)}   [{_fmt_eval_km(best_single_return_corridor_miss_ever, em_distance_km)} km]   (eval {best_single_return_corridor_miss_eval}, step {best_single_return_corridor_miss_step:,})")
    
    lines.append("")
    lines.append("BALLISTIC PROXY GEOMETRY")
    lines.append("-" * 108)
    lines.append(f"ballistic_min_rM             : mean={_fmt_eval_num(ballistic_tli_min_rM_mean,6)}  std={_fmt_eval_num(ballistic_tli_min_rM_std,6)}   [{_fmt_eval_km(ballistic_tli_min_rM_mean, em_distance_km)} km]")
    lines.append(f"ballistic_corridor_miss      : mean={_fmt_eval_num(ballistic_tli_corridor_dist_mean,6)}  std={_fmt_eval_num(ballistic_tli_corridor_dist_std,6)}   [{_fmt_eval_km(ballistic_tli_corridor_dist_mean, em_distance_km)} km]")
    lines.append(f"ballistic_corridor_hit_rate  : {_fmt_eval_num(ballistic_tli_corridor_hit_rate,3)}")
    lines.append(f"best_ballistic_min_rM_ever   : {_fmt_eval_num(best_ballistic_tli_min_rM_ever,6)}   [{_fmt_eval_km(best_ballistic_tli_min_rM_ever, em_distance_km)} km]   (eval {best_ballistic_tli_min_rM_eval}, step {best_ballistic_tli_min_rM_step:,})")
    lines.append(f"best_ballistic_miss_ever     : {_fmt_eval_num(best_ballistic_tli_corridor_dist_ever,6)}   [{_fmt_eval_km(best_ballistic_tli_corridor_dist_ever, em_distance_km)} km]   (eval {best_ballistic_tli_corridor_dist_eval}, step {best_ballistic_tli_corridor_dist_step:,})")
    lines.append(f"best_ballistic_corridor_rate : {_fmt_eval_num(best_ballistic_tli_corridor_hit_rate_ever,3)}")

    lines.append("")
    lines.append("DV / TLI")
    lines.append("-" * 108)
    lines.append(f"dv_used                      : mean={_fmt_eval_num(dv_mean,4)}  std={_fmt_eval_num(dv_std,4)}")
    lines.append(f"dv0                          : mean={_fmt_eval_num(dv0_mean,4)}  std={_fmt_eval_num(dv0_std,4)}")
    lines.append(f"tli_tau                      : mean={_fmt_eval_num(tli_tau_mean,4)}  std={_fmt_eval_num(tli_tau_std,4)}")
    lines.append(f"tli_ax                       : mean={_fmt_eval_num(tli_ax_mean,4)}  std={_fmt_eval_num(tli_ax_std,4)}")
    lines.append(f"tli_ay                       : mean={_fmt_eval_num(tli_ay_mean,4)}  std={_fmt_eval_num(tli_ay_std,4)}")
    lines.append(f"left_leo_step                : mean={_fmt_eval_num(left_leo_step_mean,2)}")

    lines.append("")
    lines.append("REWARD TERM TOTALS OVER EVAL EPISODES")
    lines.append("-" * 108)
    for k, s in reward_term_stats.items():
        lines.append(
            f"{k:<32s} mean={_fmt_eval_num(s['mean'],4):>10s}  "
            f"std={_fmt_eval_num(s['std'],4):>10s}  "
            f"min={_fmt_eval_num(s['min'],4):>10s}  "
            f"max={_fmt_eval_num(s['max'],4):>10s}"
        )

    lines.append("")
    lines.append("MISSION PROGRESSION")
    lines.append("-" * 108)
    lines.append(f"total_eval_episodes_seen     : {total_eval_episodes_seen}")
    lines.append(f"total_flyby_episodes         : {total_flyby_episodes}")
    lines.append(f"total_corridor_hit_episodes  : {total_corridor_hit_episodes}")
    lines.append(f"total_success_episodes       : {total_success_episodes}")
    lines.append(
        f"first_flyby                  : eval {first_flyby_eval}, step {first_flyby_step:,}"
        if first_flyby_step >= 0 else
        "first_flyby                  : not yet"
    )
    lines.append(
        f"first_corridor_hit           : eval {first_corridor_hit_eval}, step {first_corridor_hit_step:,}"
        if first_corridor_hit_step >= 0 else
        "first_corridor_hit           : not yet"
    )
    lines.append(
        f"first_success                : eval {first_success_eval}, step {first_success_step:,}"
        if first_success_step >= 0 else
        "first_success                : not yet"
    )

    lines.append("=" * 108)
    return "\n".join(lines)


def save_eval_summary_txt(out_dir: Path, stem: str, txt: str) -> Path:
    out_path = out_dir / f"{stem}_eval_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return out_path


def run_spawn_theta_sweep(model, eval_env, deterministic=True, n_cases=8):
    cfg = eval_env.cfg

    if bool(cfg.spawn_theta_limit_enabled):
        a = float(cfg.spawn_theta_min)
        b = float(cfg.spawn_theta_max)
    else:
        a = 0.0
        b = 2.0 * np.pi

    if n_cases <= 1:
        thetas = np.array([0.5 * (a + b)], dtype=np.float64)
    else:
        thetas = np.linspace(a, b, n_cases)

    sweep_results = []

    for theta in thetas:
        obs, info = eval_env.reset(options={"forced_spawn_theta": float(theta)})

        done = False
        trunc = False
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        reward_sum = 0.0

        while not (done or trunc):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, r, done, trunc, info = eval_env.step(action)
            reward_sum += float(r)
            episode_start = np.array([done or trunc], dtype=bool)

        traj = np.array(getattr(eval_env, "traj", []), dtype=np.float64)
        ballistic_ref_traj = np.array(getattr(eval_env, "ballistic_ref_traj", []), dtype=np.float64)

        spawn_pos_rot = None
        if traj is not None and len(traj) > 0:
            spawn_pos_rot = traj[0, :2].copy()

        tli_pos_rot = getattr(eval_env, "tli_pos_rot", None)
        if tli_pos_rot is not None:
            tli_pos_rot = np.asarray(tli_pos_rot, dtype=np.float64).copy()

        sweep_results.append({
            "spawn_theta": float(info.get("spawn_theta", np.nan)),
            "tli_theta": float(info.get("tli_theta", np.nan)),
            "dv0": float(info.get("dv0", np.nan)),
            "tli_tau": float(info.get("tli_tau", np.nan)),
            "tli_ax": float(info.get("tli_ax", np.nan)),
            "tli_ay": float(info.get("tli_ay", np.nan)),
            "reason": str(info.get("term_reason", "")),
            "ballistic_tli_corridor_hit": bool(info.get("ballistic_tli_corridor_hit", False)),
            "success_flag_latched": bool(info.get("success", False)),
            "reward_sum": float(reward_sum),
            "traj": traj,
            "ballistic_ref_traj": ballistic_ref_traj,
            "spawn_pos_rot": spawn_pos_rot,
            "tli_pos_rot": tli_pos_rot,
        })

    return sweep_results


class TrajectoryEvalCallback:
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        n_eval_episodes: int,
        plot_every: int,
        run_dir: Path,
        plots_root: Path,
        policy_label: str = "POLICY",
    ):
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.plot_every = int(plot_every)
        self.run_dir = run_dir
        self.plots_root = plots_root
        self.num_evals = 0
        self.policy_label = str(policy_label)

        # ------------------------------------------------------------
        # Best-so-far tracking across ALL eval calls
        # ------------------------------------------------------------
        self.EM_DISTANCE_KM = 384400.0

        self.best_reward_ever = -np.inf
        self.best_reward_step = -1
        self.best_reward_eval = -1

        self.best_min_rM_ever = np.inf
        self.best_min_rM_step = -1
        self.best_min_rM_eval = -1

        self.best_moon_corridor_miss_ever = np.inf
        self.best_moon_corridor_miss_step = -1
        self.best_moon_corridor_miss_eval = -1

        self.best_return_corridor_miss_ever = np.inf
        self.best_return_corridor_miss_step = -1
        self.best_return_corridor_miss_eval = -1

        self.best_postflyby_rE_ever = np.inf
        self.best_postflyby_rE_step = -1
        self.best_postflyby_rE_eval = -1

        self.total_eval_episodes_seen = 0
        self.total_flyby_episodes = 0
        self.total_corridor_hit_episodes = 0
        self.total_success_episodes = 0

        self.first_flyby_step = -1
        self.first_flyby_eval = -1


        # ------------------------------------------------------------
        # V3 ballistic-TLI tracking
        # ------------------------------------------------------------
        self.best_ballistic_tli_reward_ever = -np.inf
        self.best_ballistic_tli_reward_step = -1
        self.best_ballistic_tli_reward_eval = -1

        self.best_ballistic_tli_min_rM_ever = np.inf
        self.best_ballistic_tli_min_rM_step = -1
        self.best_ballistic_tli_min_rM_eval = -1

        self.best_ballistic_tli_corridor_dist_ever = np.inf
        self.best_ballistic_tli_corridor_dist_step = -1
        self.best_ballistic_tli_corridor_dist_eval = -1

        self.best_ballistic_tli_corridor_hit_rate_ever = -np.inf

        # ------------------------------------------------------------
        # Milestone checkpoint tracking
        # ------------------------------------------------------------
        self.best_success_rate_ever = -np.inf
        self.best_flyby_rate_ever = -np.inf
        self.best_corridor_hit_rate_ever = -np.inf
        self.best_mean_min_rM_ever = np.inf

        self.first_corridor_hit_step = -1
        self.first_corridor_hit_eval = -1

        self.first_success_step = -1
        self.first_success_eval = -1

        # ------------------------------------------------------------
        # New: single-episode milestone trackers
        # ------------------------------------------------------------
        self.best_single_ballistic_tli_reward_ever = -np.inf
        self.best_single_ballistic_tli_reward_step = -1
        self.best_single_ballistic_tli_reward_eval = -1

        self.best_single_return_corridor_miss_ever = np.inf
        self.best_single_return_corridor_miss_step = -1
        self.best_single_return_corridor_miss_eval = -1

        self.best_single_moon_corridor_miss_ever = np.inf
        self.best_single_moon_corridor_miss_step = -1
        self.best_single_moon_corridor_miss_eval = -1

        self.best_trajectory_success_rate_ever = -np.inf
        self.best_success_flag_rate_ever = -np.inf

        self.best_preservation_rate_ever = -np.inf
        self.best_degradation_rate_ever = np.inf
        self.best_rescue_rate_ever = -np.inf
        self.best_unchanged_bad_rate_ever = np.inf

        # ------------------------------------------------------------
        # New: batch trackers for cleaner printing
        # ------------------------------------------------------------
        self.last_eval_reward_term_stats: Dict[str, Dict[str, float]] = {}

        # Turn step-by-step action debug OFF by default during eval
        self.eval_action_debug = False

        self.eval_history: List[Dict[str, Any]] = []
        self.ppo_history: List[Dict[str, Any]] = []
        self.last_eval_results: List[Dict[str, Any]] = []
        self.last_eval_step: int = -1

    def _to_km(self, x: float) -> float:
        if not np.isfinite(x):
            return np.nan
        return float(x) * self.EM_DISTANCE_KM
    
    def _append_histories(
        self,
        model,
        step_count: int,
        reward_mean: float,
        preservation_rate: float,
        degradation_rate: float,
        rescue_rate: float,
        unchanged_bad_rate: float,
        dv_mean: float = np.nan,
        dv_std: float = np.nan,
        success_rate: float = np.nan,
        ballistic_success_rate: float = np.nan,
        trajectory_success_rate: float = np.nan,
        success_count: int = 0,
        ballistic_success_count: int = 0,
        trajectory_success_count: int = 0,
        n_eval_episodes: int = 0,
    ) -> None:
        self.eval_history.append({
            "eval_idx": int(self.num_evals),
            "step": int(step_count),
            "reward_mean": float(reward_mean),
            "dv_mean": float(dv_mean),
            "dv_std": float(dv_std),
            "preservation_rate": float(preservation_rate),
            "degradation_rate": float(degradation_rate),
            "rescue_rate": float(rescue_rate),
            "unchanged_bad_rate": float(unchanged_bad_rate),
            "success_rate": float(success_rate),
            "ballistic_success_rate": float(ballistic_success_rate),
            "trajectory_success_rate": float(trajectory_success_rate),
            "success_count": int(success_count),
            "ballistic_success_count": int(ballistic_success_count),
            "trajectory_success_count": int(trajectory_success_count),
            "n_eval_episodes": int(n_eval_episodes),
        })

        train_metrics = getattr(model, "last_train_metrics", None)
        if not isinstance(train_metrics, dict):
            return

        row = {
            "step": int(step_count),
            "eval_idx": int(self.num_evals),
        }
        row.update(train_metrics)

        # only append if at least one PPO metric is finite
        metric_keys = [
            "approx_kl",
            "clip_fraction",
            "clip_range",
            "policy_gradient_loss",
            "value_loss",
            "loss",
            "entropy_loss",
            "explained_variance",
            "std",
            "learning_rate",
        ]

        found_any = False
        for k in metric_keys:
            v = row.get(k, np.nan)
            if np.isfinite(v):
                found_any = True
                break

        if found_any:
            self.ppo_history.append(row)

    def save_final_training_plots(self, out_dir: Optional[Path] = None) -> None:
        if out_dir is None:
            out_dir = ensure_dir(self.run_dir / "final_training_plots")
        else:
            out_dir = ensure_dir(out_dir)

        plot_training_eval_reward_curve(
            history=self.eval_history,
            title="Mean eval reward over training",
            out_path=str(out_dir / "final_mean_eval_reward.png"),
        )

        plot_free_return_rates_curve(
            history=self.eval_history,
            title="Preservation / degradation / rescue / unchanged-bad rates",
            out_path=str(out_dir / "final_free_return_rates.png"),
        )

        plot_ppo_metrics_curve(
            history=self.ppo_history,
            title="PPO training metrics over training",
            out_path=str(out_dir / "final_ppo_metrics.png"),
        )

        save_training_history_npz(
            eval_history=self.eval_history,
            ppo_history=self.ppo_history,
            out_dir=out_dir,
            stem="final_training_curves",
        )
        
        plot_mean_eval_dv_curve(
            history=self.eval_history,
            title="Mean eval delta-v over training",
            out_path=str(out_dir / "final_mean_eval_dv.png"),
        )




    def maybe_eval(self, model, step_count: int):
        if step_count % self.eval_freq != 0:
            return

        self.num_evals += 1
        n = self.n_eval_episodes

        eval_results: List[Dict[str, Any]] = []
        reasons: Dict[str, int] = {}

        for _ in range(n):
            ep = run_eval_episode_collect(
                model=model,
                env=self.eval_env,
                deterministic=True,
                action_debug=self.eval_action_debug,
            )
            eval_results.append(ep)

            # ------------------------------------------------------------
            # DEBUG: verify deterministic reset / first action consistency
            # ------------------------------------------------------------
            reset_states = np.asarray([ep["reset_debug"]["reset_state"] for ep in eval_results], dtype=np.float64)
            reset_obs = np.asarray([ep["reset_debug"]["reset_obs"] for ep in eval_results], dtype=np.float64)
            first_actions = np.asarray([ep["first_action"] for ep in eval_results], dtype=np.float64)

            scenario_indices = [ep["reset_debug"]["scenario_index"] for ep in eval_results]
            scenario_rows = [ep["reset_debug"]["scenario_row_index"] for ep in eval_results]
            scenario_labels = [ep["reset_debug"]["scenario_label"] for ep in eval_results]
            noise_pos = [ep["reset_debug"]["scenario_noise_pos_sigma"] for ep in eval_results]
            noise_vel = [ep["reset_debug"]["scenario_noise_vel_sigma"] for ep in eval_results]

            state_spread = float(np.nanmax(np.ptp(reset_states, axis=0))) if len(reset_states) > 0 else np.nan
            obs_spread = float(np.nanmax(np.ptp(reset_obs, axis=0))) if len(reset_obs) > 0 else np.nan
            action_spread = float(np.nanmax(np.ptp(first_actions, axis=0))) if len(first_actions) > 0 else np.nan

            #print("\n[DETERMINISM DEBUG]")
            #print(f"  scenario_indices unique : {sorted(set(scenario_indices))}")
            #print(f"  scenario_rows unique    : {sorted(set(scenario_rows))}")
            #print(f"  scenario_labels unique  : {sorted(set(scenario_labels))}")
            #print(f"  noise_pos unique        : {sorted(set(noise_pos))}")
            #print(f"  noise_vel unique        : {sorted(set(noise_vel))}")
            #print(f"  max reset_state spread  : {state_spread:.12e}")
            #print(f"  max reset_obs spread    : {obs_spread:.12e}")
            #print(f"  max first_action spread : {action_spread:.12e}")
            #print("[/DETERMINISM DEBUG]\n")

            reason = str(ep["reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
            self.total_eval_episodes_seen += 1

            if ep["flyby_done"]:
                self.total_flyby_episodes += 1
                if self.first_flyby_step < 0:
                    self.first_flyby_step = int(step_count)
                    self.first_flyby_eval = int(self.num_evals)

            if ep["corridor_hit"]:
                self.total_corridor_hit_episodes += 1
                if self.first_corridor_hit_step < 0:
                    self.first_corridor_hit_step = int(step_count)
                    self.first_corridor_hit_eval = int(self.num_evals)

            trainer_mode = str(getattr(self.eval_env.cfg, "trainer_mode", "")).lower()
            tli_success_metric_enabled = bool(
                trainer_mode == "ppo_a"
                or getattr(self.eval_env.cfg, "tli_only_mode", False)
                or getattr(self.eval_env.cfg, "reward_after_tli_ballistic_enabled", False)
            )

            primary_success = (
                bool(ep["ballistic_success"])
                if tli_success_metric_enabled
                else bool(ep["trajectory_success"])
            )

            if primary_success:
                self.total_success_episodes += 1
                if self.first_success_step < 0:
                    self.first_success_step = int(step_count)
                    self.first_success_eval = int(self.num_evals)

        # ------------------------------------------------------------
        # Build batch arrays
        # ------------------------------------------------------------
        ep_rewards = [ep["reward_sum"] for ep in eval_results]

        # success_rate = ballistic initial-condition success rate
        trainer_mode = str(getattr(self.eval_env.cfg, "trainer_mode", "")).lower()
        tli_success_metric_enabled = bool(
            trainer_mode == "ppo_a"
            or getattr(self.eval_env.cfg, "tli_only_mode", False)
            or getattr(self.eval_env.cfg, "reward_after_tli_ballistic_enabled", False)
        )

        ep_ballistic_success = [1.0 if ep["ballistic_success"] else 0.0 for ep in eval_results]
        ep_trajectory_success = [1.0 if ep["trajectory_success"] else 0.0 for ep in eval_results]
        ep_success_flag = [1.0 if ep["success_flag_latched"] else 0.0 for ep in eval_results]

        # Primary success metric used for checkpoint names and headline logging.
        if tli_success_metric_enabled:
            ep_success = ep_ballistic_success
            success_metric_name = "ballistic_tli_success"
        else:
            ep_success = ep_trajectory_success
            success_metric_name = "trajectory_success"

        ep_flyby = [1.0 if ep["flyby_done"] else 0.0 for ep in eval_results]
        ep_corridor = [1.0 if ep["corridor_hit"] else 0.0 for ep in eval_results]

        ep_dv_used = [ep["dv_used"] for ep in eval_results]
        ep_dv0 = [ep["dv0"] for ep in eval_results]

        ep_min_rM = [ep["min_rM"] for ep in eval_results]
        ep_min_rE = [ep["min_rE"] for ep in eval_results]
        ep_min_rE_postflyby = [ep["min_rE_postflyby"] for ep in eval_results]
        ep_moon_corridor_miss = [ep["moon_corridor_miss"] for ep in eval_results]
        ep_return_corridor_miss = [ep["return_corridor_miss"] for ep in eval_results]
        ep_vrel_at_min_rM = [ep["vrel_at_min_rM"] for ep in eval_results]

        ep_left_leo = [1.0 if ep["left_leo"] else 0.0 for ep in eval_results]
        ep_left_leo_step = [ep["left_leo_step"] for ep in eval_results]

        ep_tli_tau = [ep["tli_tau"] for ep in eval_results]
        ep_tli_ax = [ep["tli_ax"] for ep in eval_results]
        ep_tli_ay = [ep["tli_ay"] for ep in eval_results]

        ep_ballistic_reward = [ep["ballistic_tli_reward"] for ep in eval_results]
        ep_ballistic_min_rM = [ep["ballistic_tli_min_rM"] for ep in eval_results]
        ep_ballistic_corridor_dist = [ep["ballistic_tli_corridor_dist"] for ep in eval_results]
        ep_ballistic_corridor_hit = [1.0 if ep["ballistic_tli_corridor_hit"] else 0.0 for ep in eval_results]

        ep_preserved = [
            1.0 if (ep["ballistic_success"] and ep["trajectory_success"]) else 0.0
            for ep in eval_results
        ]
        ep_degraded = [
            1.0 if (ep["ballistic_success"] and (not ep["trajectory_success"])) else 0.0
            for ep in eval_results
        ]
        ep_rescued = [
            1.0 if ((not ep["ballistic_success"]) and ep["trajectory_success"]) else 0.0
            for ep in eval_results
        ]
        ep_unchanged_bad = [
            1.0 if ((not ep["ballistic_success"]) and (not ep["trajectory_success"])) else 0.0
            for ep in eval_results
        ]

        # ------------------------------------------------------------
        # Batch summary stats
        # ------------------------------------------------------------
        sr = safe_mean(ep_success)
        trajectory_success_rate = safe_mean(ep_trajectory_success)
        success_flag_rate = safe_mean(ep_success_flag)
        flyby_rate = safe_mean(ep_flyby)
        corridor_hit_rate = safe_mean(ep_corridor)

        reward_mean = safe_mean(ep_rewards)
        reward_std = safe_std(ep_rewards)

        dv_mean = safe_mean(ep_dv_used)
        dv_std = safe_std(ep_dv_used)
        dv0_mean = safe_mean(ep_dv0)
        dv0_std = safe_std(ep_dv0)

        minrM_mean = safe_mean(ep_min_rM)
        minrM_std = safe_std(ep_min_rM)

        minrE_mean = safe_mean(ep_min_rE)
        minrE_std = safe_std(ep_min_rE)

        minrE_postflyby_mean = safe_mean(ep_min_rE_postflyby)
        minrE_postflyby_std = safe_std(ep_min_rE_postflyby)

        moon_corridor_miss_mean = safe_mean(ep_moon_corridor_miss)
        moon_corridor_miss_std = safe_std(ep_moon_corridor_miss)

        return_corridor_miss_mean = safe_mean(ep_return_corridor_miss)
        return_corridor_miss_std = safe_std(ep_return_corridor_miss)

        vrel_min_mean = safe_mean(ep_vrel_at_min_rM)
        vrel_min_std = safe_std(ep_vrel_at_min_rM)

        left_leo_rate = safe_mean(ep_left_leo)
        left_leo_step_mean = safe_mean(ep_left_leo_step)

        tli_tau_mean, tli_tau_std = safe_mean(ep_tli_tau), safe_std(ep_tli_tau)
        tli_ax_mean, tli_ax_std = safe_mean(ep_tli_ax), safe_std(ep_tli_ax)
        tli_ay_mean, tli_ay_std = safe_mean(ep_tli_ay), safe_std(ep_tli_ay)

        ballistic_tli_reward_mean = safe_mean(ep_ballistic_reward)
        ballistic_tli_reward_std = safe_std(ep_ballistic_reward)
        ballistic_tli_min_rM_mean = safe_mean(ep_ballistic_min_rM)
        ballistic_tli_min_rM_std = safe_std(ep_ballistic_min_rM)
        ballistic_tli_corridor_dist_mean = safe_mean(ep_ballistic_corridor_dist)
        ballistic_tli_corridor_dist_std = safe_std(ep_ballistic_corridor_dist)
        ballistic_tli_corridor_hit_rate = safe_mean(ep_ballistic_corridor_hit)

        preservation_rate = safe_mean(ep_preserved)
        degradation_rate = safe_mean(ep_degraded)
        rescue_rate = safe_mean(ep_rescued)
        unchanged_bad_rate = safe_mean(ep_unchanged_bad)

        # ------------------------------------------------------------
        # Best single episode from this eval batch
        # ------------------------------------------------------------
        best_reward_ep = max(eval_results, key=lambda ep: ep["reward_sum"])
        best_ballistic_ep = max(
            eval_results,
            key=lambda ep: ep["ballistic_tli_reward"] if np.isfinite(ep["ballistic_tli_reward"]) else -np.inf
        )
        best_moon_miss_ep = min(
            eval_results,
            key=lambda ep: ep["moon_corridor_miss"] if np.isfinite(ep["moon_corridor_miss"]) else np.inf
        )
        best_return_miss_ep = min(
            eval_results,
            key=lambda ep: ep["return_corridor_miss"] if np.isfinite(ep["return_corridor_miss"]) else np.inf
        )

        # ------------------------------------------------------------
        # Best-so-far updates
        # ------------------------------------------------------------
        if np.isfinite(best_reward_ep["reward_sum"]) and best_reward_ep["reward_sum"] > self.best_reward_ever:
            self.best_reward_ever = float(best_reward_ep["reward_sum"])
            self.best_reward_step = int(step_count)
            self.best_reward_eval = int(self.num_evals)
            print(f"[MILESTONE] New best single episode reward: {best_reward_ep['reward_sum']:.3f} at step {step_count:,}")

        best_min_rM_batch = safe_min(ep_min_rM)
        if np.isfinite(best_min_rM_batch) and best_min_rM_batch < self.best_min_rM_ever:
            self.best_min_rM_ever = float(best_min_rM_batch)
            self.best_min_rM_step = int(step_count)
            self.best_min_rM_eval = int(self.num_evals)

        best_moon_miss_batch = safe_min(ep_moon_corridor_miss)
        if np.isfinite(best_moon_miss_batch) and best_moon_miss_batch < self.best_moon_corridor_miss_ever:
            self.best_moon_corridor_miss_ever = float(best_moon_miss_batch)
            self.best_moon_corridor_miss_step = int(step_count)
            self.best_moon_corridor_miss_eval = int(self.num_evals)

        best_return_miss_batch = safe_min(ep_return_corridor_miss)
        if np.isfinite(best_return_miss_batch) and best_return_miss_batch < self.best_return_corridor_miss_ever:
            self.best_return_corridor_miss_ever = float(best_return_miss_batch)
            self.best_return_corridor_miss_step = int(step_count)
            self.best_return_corridor_miss_eval = int(self.num_evals)

        best_postflyby_rE_batch = safe_min(ep_min_rE_postflyby)
        if np.isfinite(best_postflyby_rE_batch) and best_postflyby_rE_batch < self.best_postflyby_rE_ever:
            self.best_postflyby_rE_ever = float(best_postflyby_rE_batch)
            self.best_postflyby_rE_step = int(step_count)
            self.best_postflyby_rE_eval = int(self.num_evals)

        if np.isfinite(ballistic_tli_min_rM_mean) and ballistic_tli_min_rM_mean < self.best_ballistic_tli_min_rM_ever:
            self.best_ballistic_tli_min_rM_ever = float(ballistic_tli_min_rM_mean)
            self.best_ballistic_tli_min_rM_step = int(step_count)
            self.best_ballistic_tli_min_rM_eval = int(self.num_evals)

        if np.isfinite(ballistic_tli_corridor_dist_mean) and ballistic_tli_corridor_dist_mean < self.best_ballistic_tli_corridor_dist_ever:
            self.best_ballistic_tli_corridor_dist_ever = float(ballistic_tli_corridor_dist_mean)
            self.best_ballistic_tli_corridor_dist_step = int(step_count)
            self.best_ballistic_tli_corridor_dist_eval = int(self.num_evals)

        # ------------------------------------------------------------
        # New reward-term batch stats
        # ------------------------------------------------------------
        reward_term_stats = aggregate_reward_term_stats(eval_results)
        self.last_eval_reward_term_stats = reward_term_stats

        self.last_eval_results = eval_results
        self.last_eval_step = int(step_count)

        self._append_histories(
            model=model,
            step_count=step_count,
            reward_mean=reward_mean,
            dv_mean=dv_mean,
            dv_std=dv_std,
            preservation_rate=preservation_rate,
            degradation_rate=degradation_rate,
            rescue_rate=rescue_rate,
            unchanged_bad_rate=unchanged_bad_rate,
            success_rate=sr,
            ballistic_success_rate=safe_mean(ep_ballistic_success),
            trajectory_success_rate=trajectory_success_rate,
            success_count=int(np.sum(ep_success)),
            ballistic_success_count=int(np.sum(ep_ballistic_success)),
            trajectory_success_count=int(np.sum(ep_trajectory_success)),
            n_eval_episodes=int(n),
        )

        self.save_final_training_plots()


        # ------------------------------------------------------------
        # Clean eval print
        # ------------------------------------------------------------
        eval_summary_txt = build_eval_summary_text(
            eval_idx=self.num_evals,
            step_count=step_count,
            n_episodes=n,
            reasons=reasons,

            sr=sr,
            trajectory_success_rate=trajectory_success_rate,
            flyby_rate=flyby_rate,
            corridor_hit_rate=corridor_hit_rate,
            success_flag_rate=success_flag_rate,
            best_trajectory_success_rate_ever=self.best_trajectory_success_rate_ever,
            best_success_flag_rate_ever=self.best_success_flag_rate_ever,

            preservation_rate=preservation_rate,
            degradation_rate=degradation_rate,
            rescue_rate=rescue_rate,
            unchanged_bad_rate=unchanged_bad_rate,

            best_preservation_rate_ever=self.best_preservation_rate_ever,
            best_degradation_rate_ever=self.best_degradation_rate_ever,
            best_rescue_rate_ever=self.best_rescue_rate_ever,
            best_unchanged_bad_rate_ever=self.best_unchanged_bad_rate_ever,
            left_leo_rate=left_leo_rate,
            left_leo_step_mean=left_leo_step_mean,

            best_single_moon_corridor_miss_ever=self.best_single_moon_corridor_miss_ever,
            best_single_moon_corridor_miss_eval=self.best_single_moon_corridor_miss_eval,
            best_single_moon_corridor_miss_step=self.best_single_moon_corridor_miss_step,

            best_single_return_corridor_miss_ever=self.best_single_return_corridor_miss_ever,
            best_single_return_corridor_miss_eval=self.best_single_return_corridor_miss_eval,
            best_single_return_corridor_miss_step=self.best_single_return_corridor_miss_step,

            reward_mean=reward_mean,
            reward_std=reward_std,
            best_reward_ever=self.best_reward_ever,
            best_reward_eval=self.best_reward_eval,
            best_reward_step=self.best_reward_step,

            dv_mean=dv_mean,
            dv_std=dv_std,
            dv0_mean=dv0_mean,
            dv0_std=dv0_std,

            minrM_mean=minrM_mean,
            minrM_std=minrM_std,
            vrel_min_mean=vrel_min_mean,
            vrel_min_std=vrel_min_std,
            best_min_rM_ever=self.best_min_rM_ever,
            best_min_rM_eval=self.best_min_rM_eval,
            best_min_rM_step=self.best_min_rM_step,
            best_moon_corridor_miss_ever=self.best_moon_corridor_miss_ever,
            best_moon_corridor_miss_eval=self.best_moon_corridor_miss_eval,
            best_moon_corridor_miss_step=self.best_moon_corridor_miss_step,

            moon_corridor_miss_mean=moon_corridor_miss_mean,
            moon_corridor_miss_std=moon_corridor_miss_std,

            minrE_mean=minrE_mean,
            minrE_std=minrE_std,
            minrE_postflyby_mean=minrE_postflyby_mean,
            minrE_postflyby_std=minrE_postflyby_std,
            return_corridor_miss_mean=return_corridor_miss_mean,
            return_corridor_miss_std=return_corridor_miss_std,
            best_return_corridor_miss_ever=self.best_return_corridor_miss_ever,
            best_return_corridor_miss_eval=self.best_return_corridor_miss_eval,
            best_return_corridor_miss_step=self.best_return_corridor_miss_step,
            best_postflyby_rE_ever=self.best_postflyby_rE_ever,
            best_postflyby_rE_eval=self.best_postflyby_rE_eval,
            best_postflyby_rE_step=self.best_postflyby_rE_step,

            ballistic_tli_reward_mean=ballistic_tli_reward_mean,
            ballistic_tli_reward_std=ballistic_tli_reward_std,
            best_ballistic_tli_reward_ever=self.best_ballistic_tli_reward_ever,
            best_ballistic_tli_reward_eval=self.best_ballistic_tli_reward_eval,
            best_ballistic_tli_reward_step=self.best_ballistic_tli_reward_step,
            best_single_ballistic_tli_reward_ever=self.best_single_ballistic_tli_reward_ever,
            best_single_ballistic_tli_reward_eval=self.best_single_ballistic_tli_reward_eval,
            best_single_ballistic_tli_reward_step=self.best_single_ballistic_tli_reward_step,

            ballistic_tli_min_rM_mean=ballistic_tli_min_rM_mean,
            ballistic_tli_min_rM_std=ballistic_tli_min_rM_std,
            ballistic_tli_corridor_dist_mean=ballistic_tli_corridor_dist_mean,
            ballistic_tli_corridor_dist_std=ballistic_tli_corridor_dist_std,
            ballistic_tli_corridor_hit_rate=ballistic_tli_corridor_hit_rate,
            best_ballistic_tli_min_rM_ever=self.best_ballistic_tli_min_rM_ever,
            best_ballistic_tli_min_rM_eval=self.best_ballistic_tli_min_rM_eval,
            best_ballistic_tli_min_rM_step=self.best_ballistic_tli_min_rM_step,
            best_ballistic_tli_corridor_dist_ever=self.best_ballistic_tli_corridor_dist_ever,
            best_ballistic_tli_corridor_dist_eval=self.best_ballistic_tli_corridor_dist_eval,
            best_ballistic_tli_corridor_dist_step=self.best_ballistic_tli_corridor_dist_step,
            best_ballistic_tli_corridor_hit_rate_ever=self.best_ballistic_tli_corridor_hit_rate_ever,

            best_success_rate_ever=self.best_success_rate_ever,
            best_flyby_rate_ever=self.best_flyby_rate_ever,
            best_corridor_hit_rate_ever=self.best_corridor_hit_rate_ever,

            total_eval_episodes_seen=self.total_eval_episodes_seen,
            total_flyby_episodes=self.total_flyby_episodes,
            total_corridor_hit_episodes=self.total_corridor_hit_episodes,
            total_success_episodes=self.total_success_episodes,
            first_flyby_eval=self.first_flyby_eval,
            first_flyby_step=self.first_flyby_step,
            first_corridor_hit_eval=self.first_corridor_hit_eval,
            first_corridor_hit_step=self.first_corridor_hit_step,
            first_success_eval=self.first_success_eval,
            first_success_step=self.first_success_step,

            tli_tau_mean=tli_tau_mean,
            tli_tau_std=tli_tau_std,
            tli_ax_mean=tli_ax_mean,
            tli_ax_std=tli_ax_std,
            tli_ay_mean=tli_ay_mean,
            tli_ay_std=tli_ay_std,

            reward_term_stats=reward_term_stats,
            em_distance_km=self.EM_DISTANCE_KM,
        )

        print("\n" + eval_summary_txt + "\n")

        # Always keep a latest-eval save
        stage_idx = infer_curriculum_stage_from_step(self.curriculum, int(step_count))

        trainer_mode = str(getattr(self.eval_env.cfg, "trainer_mode", "")).lower()

        if trainer_mode == "ppo_a":
            save_lunar_distance = ballistic_tli_min_rM_mean
            save_corridor_miss = ballistic_tli_corridor_dist_mean
        else:
            save_lunar_distance = minrM_mean
            save_corridor_miss = return_corridor_miss_mean

        save_eval_model_with_stats(
            model=model,
            run_dir=self.run_dir,
            stage_idx=stage_idx,
            step_count=int(step_count),
            reward_mean=reward_mean,
            success_rate=sr,
            lunar_distance=save_lunar_distance,
            corridor_miss=save_corridor_miss,
            policy_label="Model",
        )

        # ------------------------------------------------------------
        # Keep your plotting/report behavior
        # ------------------------------------------------------------
        if (self.num_evals % self.plot_every == 0):
            traj_dir = ensure_dir(self.plots_root / f"trajectories_{timestamp_str()}")

            base = f"eval{self.num_evals:04d}_step{step_count:09d}"

            save_eval_summary_txt(
                out_dir=traj_dir,
                stem=base,
                txt=eval_summary_txt,
            )

            plot_eval_trajectories_grid(
                cfg=self.eval_env.cfg,
                eval_results=eval_results,
                title=f"All eval trajectories ({base})",
                out_path=str(traj_dir / f"traj_grid_{base}.png"),
                ncols=4,
            )

            # Pick one ACTUAL eval episode from this batch at random
            success_eps = [ep for ep in eval_results if ep["success_strict"]]
            flyby_eps = [ep for ep in eval_results if ep["flyby_done"]]

            if len(success_eps) > 0:
                pool = success_eps
            elif len(flyby_eps) > 0:
                pool = flyby_eps
            else:
                pool = eval_results

            plot_ep = pool[np.random.randint(len(pool))]

            rewards = np.asarray(plot_ep["rewards"], dtype=np.float64)
            terms_ts = plot_ep["terms_ts"]
            info_last = plot_ep["info_last"]
            reward_records = plot_ep.get("reward_records", None)
            reward_audit = plot_ep.get("reward_audit", None)


            traj = np.asarray(plot_ep["traj"], dtype=np.float64)
            t_hist = np.asarray(plot_ep["t_hist"], dtype=np.float64)
            burns = plot_ep["burns"]
            burn_events = plot_ep["burn_events"]
            ballistic_ref_traj = plot_ep["ballistic_ref_traj"]
            ballistic_ref_t_hist = plot_ep["ballistic_ref_t_hist"]
            mcc_ballistic_overlays = plot_ep["mcc_ballistic_overlays"]

            if traj is not None and len(traj) > 0:
                save_episode_report_txt(
                    out_dir=traj_dir,
                    stem=base,
                    env=plot_ep["env_snapshot"] if plot_ep.get("env_snapshot", None) is not None else self.eval_env,
                    rewards=rewards,
                    terms_ts=terms_ts,
                    info_last=info_last,
                    reward_records=reward_records,
                    audit=reward_audit,
                )

                plot_trajectory(
                    plot_ep["env_snapshot"].cfg,
                    traj,
                    burns=burns,
                    burn_events=burn_events,
                    ballistic_ref_traj=ballistic_ref_traj,
                    ballistic_terminal_marker=plot_ep.get("ballistic_terminal_marker_rot", None),
                    terminal_marker=plot_ep.get("terminal_marker_rot", None),
                    title=f"Rotating frame ({base})",
                    out_path=str(traj_dir / f"traj_rot_{base}.png")
                )
                if bool(getattr(RUN, "generate_mcc_eval_plot", False)):
                    plot_trajectory_mcc_debug(
                        plot_ep["env_snapshot"].cfg,
                        traj,
                        burn_events=burn_events,
                        ballistic_ref_traj=ballistic_ref_traj,
                        ballistic_terminal_marker=plot_ep.get("ballistic_terminal_marker_rot", None),
                        terminal_marker=plot_ep.get("terminal_marker_rot", None),
                        mcc_ballistic_overlays=mcc_ballistic_overlays,
                        title=f"MCC debug ({base})",
                        out_path=str(traj_dir / f"traj_mcc_debug_{base}.png")
                    )

                plot_trajectory_earth_centered_inertial(
                    plot_ep["env_snapshot"].cfg,
                    traj,
                    t_hist,
                    ballistic_ref_traj=ballistic_ref_traj,
                    ballistic_ref_t_hist=ballistic_ref_t_hist,
                    title=f"Inertial ({base})",
                    out_path=str(traj_dir / f"traj_inert_{base}.png")
                )
                # --------------------------------------------------
                # SAVE REUSABLE EPISODE ARCHIVE (.npz + .json)
                # --------------------------------------------------
                save_eval_episode_archive_npz_json(
                    ep=plot_ep,
                    out_dir=traj_dir,
                    stem=base,
                    clip_radius=1.5,
                )

                print(f"[EXPORT] Saved episode archive: {traj_dir / f'{base}_arrays.npz'}")

                if (
                    bool(getattr(RUN, "plot_spawn_sweep_enabled", False))
                    and str(getattr(self.eval_env.cfg, "trainer_mode", "ppo_a")).lower() == "ppo_a"
                ):
                    every = max(1, int(getattr(RUN, "plot_spawn_sweep_every_evals", 1)))
                    if (self.num_evals % every) == 0:
                        sweep_results = run_spawn_theta_sweep(
                            model=model,
                            eval_env=self.eval_env,
                            deterministic=bool(getattr(RUN, "plot_spawn_sweep_deterministic", True)),
                            n_cases=int(getattr(RUN, "plot_spawn_sweep_count", 8)),
                        )

                        plot_spawn_theta_sweep(
                            cfg=self.eval_env.cfg,
                            sweep_results=sweep_results,
                            title=f"Theta sweep ({base})",
                            out_path=str(traj_dir / f"theta_sweep_{base}.png"),
                        )

                        save_spawn_theta_sweep_txt(
                            out_dir=traj_dir,
                            stem=base,
                            sweep_results=sweep_results,
                        )





def build_train_and_eval(
    base_cfg: CR3BPConfig,
    reward_cfg: RewardConfig,
    stage: CurriculumStage,
):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    stage_cfg = apply_stage_to_cfg(base_cfg, stage)
    reward_factory = build_reward_factory(reward_cfg, stage.reward_weights)

    env_fns = []
    for i in range(RUN.n_envs):
        def make_one(i=i):
            rm = reward_factory()
            cfg_i = CR3BPConfig(**vars(stage_cfg))
            return Monitor(CR3BPFreeReturnEnv(cfg_i, seed=RUN.train_seed + i, reward_model=rm))
        env_fns.append(make_one)

    train_env = DummyVecEnv(env_fns)
    eval_env = CR3BPFreeReturnEnv(
        CR3BPConfig(**vars(stage_cfg)),
        seed=RUN.eval_seed,
        reward_model=reward_factory(),
    )

    eval_env.set_debug_eval(True)
    print("\n[BUILD TRAIN/EVAL]")
    print(f"  trainer_mode      = {stage_cfg.trainer_mode}")
    print(f"  policy_family     = {cfg_policy_label(stage_cfg)}")
    print(f"  tli_control_mode  = {stage_cfg.tli_control_mode}")
    return train_env, eval_env, stage_cfg


def get_model_backend_info(model) -> dict:
    """
    Inspect the ACTUAL instantiated model and rollout buffer.
    This avoids manual version strings and reports what Python really loaded.
    """
    ppo_cls = type(model)
    ppo_module = inspect.getmodule(ppo_cls)
    ppo_file = inspect.getfile(ppo_cls)

    rb_cls = type(model.rollout_buffer)
    rb_module = inspect.getmodule(rb_cls)
    rb_file = inspect.getfile(rb_cls)

    return {
        "ppo_class_name": ppo_cls.__name__,
        "ppo_module_name": ppo_module.__name__ if ppo_module is not None else "UNKNOWN_MODULE",
        "ppo_file": str(ppo_file),
        "buffer_class_name": rb_cls.__name__,
        "buffer_module_name": rb_module.__name__ if rb_module is not None else "UNKNOWN_MODULE",
        "buffer_file": str(rb_file),
    }


def build_model(
    train_env,
    out_dir: Path,
    ent_coef: float,
    lstm_hidden_size: int = 128,
    pi_layers: Tuple[int, ...] = (128, 128),
    vf_layers: Tuple[int, ...] = (128, 128),
):
    from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2

    model = TimeAwareRecurrentPPOv2(
        policy="SquashedMlpLstmPolicy",
        env=train_env,
        verbose=1,
        gamma=RUN.gamma,
        gae_lambda=RUN.gae_lambda,
        n_steps=RUN.n_steps,
        batch_size=RUN.batch_size,
        n_epochs=RUN.n_epochs,
        learning_rate=RUN.learning_rate,
        ent_coef=float(ent_coef),
        clip_range=RUN.clip_range,
        max_grad_norm=RUN.max_grad_norm,
        policy_kwargs=dict(
            lstm_hidden_size=int(lstm_hidden_size),
            n_lstm_layers=1,
            enable_critic_lstm=True,
            net_arch=dict(
                pi=list(pi_layers),
                vf=list(vf_layers),
            ),
        ),
        tensorboard_log=str(out_dir / "tb"),
        device=RUN.device,
    )
    backend_info = get_model_backend_info(model)

    print("\n===== ACTUAL RL BACKEND IN USE =====")
    print(f"PPO class      : {backend_info['ppo_class_name']}")
    print(f"PPO module     : {backend_info['ppo_module_name']}")
    print(f"PPO file       : {backend_info['ppo_file']}")
    print(f"Buffer class   : {backend_info['buffer_class_name']}")
    print(f"Buffer module  : {backend_info['buffer_module_name']}")
    print(f"Buffer file    : {backend_info['buffer_file']}")
    print("====================================\n")

    print("\n===== SQUASH VERIFICATION =====")
    print("policy class      :", type(model.policy).__name__)
    print("action_dist class :", type(model.policy.action_dist).__name__)
    print("use_sde           :", getattr(model.policy, "use_sde", "MISSING"))
    print("squash_output     :", getattr(model.policy, "squash_output", "MISSING"))
    print("================================\n")

    return model


from stable_baselines3.common.callbacks import BaseCallback



def extract_latest_ppo_metrics(model) -> Optional[Dict[str, float]]:
    """
    Pull the latest PPO training metrics from the model logger.

    Returns None if nothing useful is available yet.
    """
    logger = getattr(model, "logger", None)
    if logger is None:
        return None

    name_to_value = getattr(logger, "name_to_value", None)
    if not isinstance(name_to_value, dict) or len(name_to_value) == 0:
        return None

    # Stable-Baselines logger keys typically look like "train/approx_kl", etc.
    key_map = {
        "approx_kl": "train/approx_kl",
        "clip_fraction": "train/clip_fraction",
        "clip_range": "train/clip_range",
        "policy_gradient_loss": "train/policy_gradient_loss",
        "value_loss": "train/value_loss",
        "loss": "train/loss",
        "entropy_loss": "train/entropy_loss",
        "explained_variance": "train/explained_variance",
        "std": "train/std",
        "learning_rate": "train/learning_rate",
    }

    out: Dict[str, float] = {}
    found_any = False

    for out_key, logger_key in key_map.items():
        val = name_to_value.get(logger_key, None)
        if val is None:
            out[out_key] = np.nan
            continue

        try:
            fval = float(val)
        except Exception:
            fval = np.nan

        out[out_key] = fval
        if np.isfinite(fval):
            found_any = True

    if not found_any:
        return None

    return out

class EvalAndCheckpointCallback(BaseCallback):
    def __init__(
        self,
        evaluator,
        run_dir: Path,
        eval_every: int,
        checkpoint_every: int,
        start_step: int = 0,
        verbose: int = 0,
        policy_label: str = "POLICY",
    ):
        super().__init__(verbose)
        self.evaluator = evaluator
        self.run_dir = run_dir
        self.eval_every = int(eval_every)
        self.checkpoint_every = int(checkpoint_every)
        self.policy_label = str(policy_label)

        self.next_eval = int(
            max(eval_every, ((start_step // eval_every) + 1) * eval_every)
        )
        self.next_ckpt = int(
            max(checkpoint_every, ((start_step // checkpoint_every) + 1) * checkpoint_every)
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        total_done = int(self.model.num_timesteps)

        # ------------------------------------------------------------
        # NEW: capture latest PPO metrics right after this rollout/update
        # ------------------------------------------------------------
        latest_metrics = extract_latest_ppo_metrics(self.model)
        if latest_metrics is not None:
            self.model.last_train_metrics = latest_metrics

        while total_done >= self.next_ckpt:
            #save_model_timestamped(
            #    self.model,
            #    self.run_dir,
            #    tag=f"checkpoint_step_{self.next_ckpt}",
            #    policy_label=self.policy_label,
            #)
            #save_model_timestamped(
            #    self.model,
            #    self.run_dir,
            #    tag="model_latest",
            #    policy_label=self.policy_label,
            #)
            #print(f"Checkpoint saved at step {self.next_ckpt:,} into: {self.run_dir}")
            self.next_ckpt += self.checkpoint_every

        while total_done >= self.next_eval:
            self.evaluator.maybe_eval(self.model, self.next_eval)
            self.next_eval += self.eval_every

def apply_current_ppo_settings(model) -> None:
    model.n_steps = int(RUN.n_steps)
    model.batch_size = int(RUN.batch_size)
    model.n_epochs = int(RUN.n_epochs)

    # FIXED
    model.clip_range = lambda _: float(RUN.clip_range)

    model.max_grad_norm = float(RUN.max_grad_norm)
    model.gae_lambda = float(RUN.gae_lambda)
    model.gamma = float(RUN.gamma)

    # learning rate schedule
    model.learning_rate = float(RUN.learning_rate)
    model.lr_schedule = lambda _: float(RUN.learning_rate)

    print("\n[APPLY CURRENT PPO SETTINGS]")
    print(f"  lr        = {RUN.learning_rate}")
    print(f"  clip      = {RUN.clip_range}")
    print(f"  epochs    = {RUN.n_epochs}")


def build_selected_curriculum(training_profile: str):
    profile = str(training_profile).strip().lower()

    if profile in ("ppo_tli", "ppo_a", "tli"):
        return build_curriculum_ppoa(kms_to_nondim_dv)

    if profile in ("ppo_mcc", "ppo_b", "mcc"):
        return build_curriculum_ppob()

    raise ValueError(f"Unknown training profile: {training_profile}")





def train(training_profile: str, resume_policy_path: Optional[Path] = None):
    

    base_cfg = CR3BPConfig()
    reward_cfg = RewardConfig()

    base_cfg.staged_tli_cumulative_dv_target = kms_to_nondim_dv(3.01)

    curriculum, overrides = build_selected_curriculum(training_profile)

    apply_overrides(base_cfg, overrides.get("env"))
    apply_overrides(reward_cfg, overrides.get("reward"))
    apply_overrides(RUN, overrides.get("run"))
    
    snap_curriculum_timesteps(curriculum)
    total_curriculum_steps = sum(int(stage.timesteps) for stage in curriculum)
    stage0_cfg_for_label = apply_stage_to_cfg(base_cfg, curriculum[0])
    run_policy_label = cfg_policy_label(stage0_cfg_for_label)

    rollout_block = ppo_rollout_block_size()

    RUN.eval_interval_steps = round_up_to_rollout_multiple(RUN.eval_interval_steps)
    RUN.checkpoint_every = round_up_to_rollout_multiple(RUN.checkpoint_every)

    print("\nTotal curriculum steps after snapping:")
    print(f"{total_curriculum_steps:,}")
    print("")
    print(f"PPO rollout block size      : {rollout_block}")
    print(f"Synced eval_interval_steps  : {RUN.eval_interval_steps}")
    print(f"Synced checkpoint_every     : {RUN.checkpoint_every}")
    print(f"Plots are produced every    : {RUN.plot_every_evals} eval(s)")
    print(f"PRE_TLI drift min (minutes) : {RUN.drift_min_minutes_pre_tli}")
    print(f"PRE_TLI drift max (minutes) : {RUN.drift_max_minutes_pre_tli}")
    print(f"POST_TLI drift min (minutes): {RUN.drift_min_minutes_post_tli}")
    print(f"POST_TLI drift max (minutes): {RUN.drift_max_minutes_post_tli}")
    print(f"PRE_TLI burn deadzone frac  : {RUN.pre_tli_burn_deadzone_frac_of_tli_cap}")
    print(f"No-TLI terminate orbits     : {RUN.no_tli_terminate_after_leo_orbits}")

    if RUN.tli_dv_max_kms is not None:
        print(f"TLI DV cap (km/s)          : {RUN.tli_dv_max_kms}")
        print(f"TLI DV cap (nondim)        : {kms_to_nondim_dv(RUN.tli_dv_max_kms):.6f}")

    if RUN.mcc_dv_max_kms is not None:
        print(f"MCC DV cap (km/s)          : {RUN.mcc_dv_max_kms}")
        print(f"MCC DV cap (nondim)        : {kms_to_nondim_dv(RUN.mcc_dv_max_kms):.6f}")

    script_path = __file__
    run_dir = make_new_run_dir(script_path, run_label=run_policy_label)
    plots_root = ensure_dir(run_dir / f"plots_{timestamp_str()}")
    ensure_dir(run_dir / "tb")

    resume_start_step = 0
    total_done = 0

    if resume_policy_path is None:
        resume_label = "fresh training"
        resume_source_txt = "fresh training"
        loaded_policy_old_steps = None
    else:
        resume_policy_path = Path(resume_policy_path).resolve()
        if not resume_policy_path.exists():
            raise FileNotFoundError(f"Resume policy not found: {resume_policy_path}")

        resume_label = str(resume_policy_path)
        resume_source_txt = str(resume_policy_path)

        try:
            from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2
            tmp_model = TimeAwareRecurrentPPOv2.load(str(resume_policy_path), device=RUN.device)
            loaded_policy_old_steps = int(getattr(tmp_model, "num_timesteps", -1))
            del tmp_model
        except Exception:
            loaded_policy_old_steps = None

    cfg_txt_path = save_run_configuration_txt(
        run_dir,
        base_cfg,
        reward_cfg,
        curriculum,
        resume_source=resume_source_txt,
    )

    print(f"Config saved: {cfg_txt_path}")
    print(f"Training source: {resume_label}")

    with open(run_dir / "resume_info.txt", "w", encoding="utf-8") as f:
        f.write(f"training_source = {resume_label}\n")
        f.write("training_mode = restart_curriculum_from_stage0\n")
        f.write("restart_stage = 0\n")
        f.write("restart_step = 0\n")
        if loaded_policy_old_steps is not None:
            f.write(f"loaded_policy_old_steps = {loaded_policy_old_steps}\n")

    train_env, eval_env, stage0_cfg = build_train_and_eval(base_cfg, reward_cfg, curriculum[0])

    from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2
    

    if resume_policy_path is None:
        model = build_model(train_env, run_dir, ent_coef=curriculum[0].entropy_coef)
        attach_fresh_logger(model, run_dir)
        append_backend_info_to_run_config(cfg_txt_path, model)

        # Apply stage-0 manual std override if enabled
        apply_stage_log_std_override(model, curriculum[0])
    else:
        model = TimeAwareRecurrentPPOv2.load(str(resume_policy_path), env=train_env, device=RUN.device)

        backend_info = get_model_backend_info(model)
        append_backend_info_to_run_config(cfg_txt_path, model)
        print("\n===== ACTUAL RL BACKEND IN USE (LOADED MODEL) =====")
        print(f"PPO class      : {backend_info['ppo_class_name']}")
        print(f"PPO module     : {backend_info['ppo_module_name']}")
        print(f"PPO file       : {backend_info['ppo_file']}")
        print(f"Buffer class   : {backend_info['buffer_class_name']}")
        print(f"Buffer module  : {backend_info['buffer_module_name']}")
        print(f"Buffer file    : {backend_info['buffer_file']}")
        print("===================================================\n")

        attach_fresh_logger(model, run_dir)

        # IMPORTANT: override loaded checkpoint PPO settings with CURRENT script settings
        apply_current_ppo_settings(model)

        model.ent_coef = float(curriculum[0].entropy_coef)

        # Apply stage-0 manual std override if enabled
        apply_stage_log_std_override(model, curriculum[0])

        model.num_timesteps = 0
        if hasattr(model, "_num_timesteps_at_start"):
            model._num_timesteps_at_start = 0

    # Use your old TrajectoryEvalCallback unchanged here
    evaluator = TrajectoryEvalCallback(
        eval_env=eval_env,
        eval_freq=RUN.eval_interval_steps,
        n_eval_episodes=RUN.eval_episodes,
        plot_every=RUN.plot_every_evals,
        run_dir=run_dir,
        plots_root=plots_root,
        policy_label=run_policy_label,
    )

    evaluator.curriculum = curriculum

    live_callback = EvalAndCheckpointCallback(
        evaluator=evaluator,
        run_dir=run_dir,
        eval_every=RUN.eval_interval_steps,
        checkpoint_every=RUN.checkpoint_every,
        start_step=0,
        policy_label=run_policy_label,
    )

    print("\nTraining handshake")
    print("-" * 78)
    if resume_policy_path is None:
        print("mode                         : fresh training")
    else:
        print("mode                         : loaded checkpoint as base initialization")
        print(f"loaded policy path           : {resume_policy_path}")
        print(f"loaded policy old steps      : {loaded_policy_old_steps}")
    print("curriculum restart stage     : 0")
    print("curriculum restart step      : 0")
    print(f"stage 0 name                 : {curriculum[0].name}")
    print(f"stage 0 trainer_mode         : {curriculum[0].trainer_mode}")
    print(f"stage 0 policy_family        : {stage_policy_label(curriculum[0])}")
    print(f"stage 0 ppo_b_theta          : {curriculum[0].ppo_b_baseline_theta}")
    print(f"stage 0 ppo_b_ax             : {curriculum[0].ppo_b_baseline_ax}")
    print(f"stage 0 ppo_b_ay             : {curriculum[0].ppo_b_baseline_ay}")
    print(f"stage 0 ppo_b_tau            : {curriculum[0].ppo_b_baseline_tau}")
    print(f"stage 0 tli_control_mode     : {curriculum[0].tli_control_mode}")
    print(f"stage 0 entropy              : {curriculum[0].entropy_coef}")
    print(f"stage 0 mcc_enabled          : {curriculum[0].mcc_enabled}")
    print(f"stage 0 tli_only_mode        : {curriculum[0].tli_only_mode}")
    print(f"stage 0 reward_after_tli     : {curriculum[0].reward_after_tli_ballistic_enabled}")
    print("using config source          : CURRENT SCRIPT")
    print("-" * 78)

    print("\nResume scheduling")
    print("-" * 60)
    print(f"resume_start_step : {resume_start_step}")
    print(f"next eval         : {live_callback.next_eval}")
    print(f"next checkpoint   : {live_callback.next_ckpt}")
    print("-" * 60)

    

    train_start_wall = time.time()
    interrupted = False
    
    try:
        for si, stage in enumerate(curriculum):
            current_policy_label = stage_policy_label(stage)
            evaluator.policy_label = current_policy_label
            live_callback.policy_label = current_policy_label
            print("\n" + "=" * 78)
            print(f"CURRICULUM STAGE {si+1}/{len(curriculum)}: {stage.name}")
            print(f" timesteps={stage.timesteps:,} | ent_coef={stage.entropy_coef}")
            print(f" trainer_mode={stage.trainer_mode}")
            print(f" tli_control_mode={stage.tli_control_mode}")
            print(f" mcc_enabled={stage.mcc_enabled}")
            print(f" tli_only_mode={stage.tli_only_mode}")
            print(f" reward_after_tli_ballistic_enabled={stage.reward_after_tli_ballistic_enabled}")
            print(f" spawn_theta_limit_enabled={stage.spawn_theta_limit_enabled}")
            print(f" spawn_theta_min/max=[{stage.spawn_theta_min:.6f}, {stage.spawn_theta_max:.6f}]")
            print(f" ppo_b_baseline_theta={stage.ppo_b_baseline_theta}")
            print(f" ppo_b_baseline_ax={stage.ppo_b_baseline_ax}")
            print(f" ppo_b_baseline_ay={stage.ppo_b_baseline_ay}")
            print(f" ppo_b_baseline_tau={stage.ppo_b_baseline_tau}")
            print(f" ppo_b_baseline_state_noise_pos={stage.ppo_b_baseline_state_noise_pos}")
            print(f" ppo_b_baseline_state_noise_vel={stage.ppo_b_baseline_state_noise_vel}")
            print(f" ppo_b_case_source={stage.ppo_b_case_source}")
            print(f" ppo_b_library_path={stage.ppo_b_library_path}")
            print(f" ppo_b_prob_good={stage.ppo_b_prob_good}")
            print(f" ppo_b_prob_savable={stage.ppo_b_prob_savable}")
            print(f" ppo_b_prob_bad={stage.ppo_b_prob_bad}")
            print(f" ppo_b_eval_use_same_distribution={stage.ppo_b_eval_use_same_distribution}")
            print(f" ppo_b_noise_theta_deg={stage.ppo_b_noise_theta_deg}")
            print(f" ppo_b_noise_tli_dir_deg={stage.ppo_b_noise_tli_dir_deg}")
            print(f" ppo_b_noise_tli_dv_kms={stage.ppo_b_noise_tli_dv_kms}")
            print(f" dv_noise_sigma_tli={stage.dv_noise_sigma_tli}")
            print(f" dv_noise_sigma_mcc={stage.dv_noise_sigma_mcc}")
            print(f" use_manual_log_std={stage.use_manual_log_std}")
            print(f" manual_log_std_value={stage.manual_log_std_value}")
            print(f" weights={stage.reward_weights}")
            print("=" * 78 + "\n")

            if si > 0:
                tmp = run_dir / "_TEMP_STAGE_TRANSFER.zip"
                model.save(str(tmp))
                try:
                    train_env.close()
                except Exception:
                    pass

                train_env, eval_env, stage_cfg = build_train_and_eval(base_cfg, reward_cfg, stage)
                evaluator.eval_env = eval_env

                model = TimeAwareRecurrentPPOv2.load(str(tmp), env=train_env, device=RUN.device)

                apply_current_ppo_settings(model)
                attach_fresh_logger(model, run_dir)

            model.ent_coef = float(stage.entropy_coef)
            apply_stage_log_std_override(model, stage)

            remaining = int(stage.timesteps)
            chunk = 4 * rollout_block

            while remaining > 0:
                this = min(chunk, remaining)

                chunk_t0 = time.time()
                
                model.learn(
                    total_timesteps=this,
                    reset_num_timesteps=False,
                    progress_bar=True,
                    callback=live_callback,
                )
                chunk_dt = time.time() - chunk_t0

                remaining -= this
                total_done += this

                fps_est = float(this) / max(chunk_dt, 1e-9)

                to_eval = steps_until_next_multiple(total_done, RUN.eval_interval_steps)
                to_ckpt = steps_until_next_multiple(total_done, RUN.checkpoint_every)

                rollouts_done = total_done // rollout_block
                total_remaining = max(0, total_curriculum_steps - total_done)

                next_outer_print_steps = min(chunk, remaining) if remaining > 0 else 0

                eta_next_print = format_seconds_to_hms(
                    next_outer_print_steps / max(fps_est, 1e-9)
                ) if next_outer_print_steps > 0 else "done"

                eta_eval = format_seconds_to_hms(
                    to_eval / max(fps_est, 1e-9)
                ) if to_eval > 0 else "now"

                eta_ckpt = format_seconds_to_hms(
                    to_ckpt / max(fps_est, 1e-9)
                ) if to_ckpt > 0 else "now"

                eta_stage_done = format_seconds_to_hms(
                    remaining / max(fps_est, 1e-9)
                ) if remaining > 0 else "done"

                eta_total_done = format_seconds_to_hms(
                    total_remaining / max(fps_est, 1e-9)
                ) if total_remaining > 0 else "done"

                elapsed_total = format_seconds_to_hms(time.time() - train_start_wall)

                print(
                    f"[TRAIN STATUS] "
                    f"total_steps={total_done:,} | "
                    f"rollouts_done={rollouts_done:,} | "
                    f"chunk_steps={this:,} | "
                    f"fps~={fps_est:.1f} | "
                    f"elapsed={elapsed_total}"
                )
                print(
                    f"               next outer print in ~{next_outer_print_steps:,} steps ({eta_next_print}) | "
                    f"next eval in {to_eval:,} steps ({eta_eval}) | "
                    f"next checkpoint in {to_ckpt:,} steps ({eta_ckpt})"
                )
                print(
                    f"               stage remaining={remaining:,} ({eta_stage_done}) | "
                    f"total remaining={total_remaining:,} ({eta_total_done})"
                )

                try:
                    train_env.env_method("set_global_step", total_done)
                    eval_env.set_global_step(total_done)
                except Exception:
                    pass


    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPT] Ctrl+C detected. Saving latest model and final plots...")

    finally:
        final_label = locals().get("current_policy_label", run_policy_label)

        try:
            save_model_timestamped(
                model,
                run_dir,
                tag="model_interrupted" if interrupted else "model_final",
                policy_label=final_label,
            )
        except Exception as e:
            print(f"[WARN] Could not save final/interrupted model: {e}")

        try:
            evaluator.save_final_training_plots()
            print(f"[FINAL PLOTS] Saved in: {run_dir / 'final_training_plots'}")
        except Exception as e:
            print(f"[WARN] Could not save final training plots: {e}")

        try:
            if len(getattr(evaluator, "last_eval_results", [])) > 0:
                final_eval_dir = ensure_dir(run_dir / "final_training_plots" / "last_eval_snapshot")
                final_base = f"last_eval_step_{int(getattr(evaluator, 'last_eval_step', -1)):09d}"

                plot_eval_trajectories_grid(
                    cfg=evaluator.eval_env.cfg,
                    eval_results=evaluator.last_eval_results,
                    title=f"Last eval trajectories ({final_base})",
                    out_path=str(final_eval_dir / f"{final_base}_grid.png"),
                    ncols=4,
                )
        except Exception as e:
            print(f"[WARN] Could not save last eval grid: {e}")

        print(f"\nTraining finished. Outputs are in: {run_dir}")





def extract_step_from_policy_name(policy_path: Path) -> int:
    name = policy_path.stem

    tokens = name.replace("__", "_").split("_")
    for i, tok in enumerate(tokens):
        if tok == "step" and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                pass

    for tok in tokens:
        if tok.isdigit():
            return int(tok)

    return -1


def parse_run_config_txt(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "run_config.txt"
    data: Dict[str, Any] = {}

    if not cfg_path.exists():
        return data

    with open(cfg_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("=") or line.startswith("["):
                continue
            if " = " not in line:
                continue

            key, val = line.split(" = ", 1)
            key = key.strip()
            val = val.strip()

            if val.lower() in ("true", "false"):
                data[key] = (val.lower() == "true")
                continue

            try:
                if "." in val or "e" in val.lower():
                    data[key] = float(val)
                else:
                    data[key] = int(val)
                continue
            except ValueError:
                data[key] = val

    return data


def _coerce_saved_value(val: str):
    val = str(val).strip()

    if val.lower() in ("true", "false"):
        return (val.lower() == "true")

    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_saved_curriculum_stages(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Parse the [CURRICULUM] stage blocks from a saved run_config.txt.

    Returns a list like:
        [
            {
                "stage_idx": 0,
                "stage_name": "...",
                "timesteps": ...,
                "trainer_mode": ...,
                ...
            },
            ...
        ]
    """
    cfg_path = run_dir / "run_config.txt"
    stages: List[Dict[str, Any]] = []

    if not cfg_path.exists():
        return stages

    current_stage: Optional[Dict[str, Any]] = None
    in_curriculum = False

    stage_header_re = re.compile(r"^Stage\s+(\d+)\s*:\s*(.+?)\s*$")

    with open(cfg_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("[CURRICULUM]"):
                in_curriculum = True
                continue

            if (
                (stripped.startswith("[") and stripped != "[CURRICULUM]")
                or stripped.startswith("===")
            ):
                if current_stage is not None:
                    stages.append(current_stage)
                    current_stage = None
                in_curriculum = False
                continue

            if not in_curriculum:
                continue

            m = stage_header_re.match(stripped)
            if m is not None:
                if current_stage is not None:
                    stages.append(current_stage)

                stage_idx_1based = int(m.group(1))
                stage_name = str(m.group(2)).strip()

                current_stage = {
                    "stage_idx": stage_idx_1based - 1,
                    "stage_name": stage_name,
                }
                continue

            if current_stage is None:
                continue

            if " = " in stripped:
                key, val = stripped.split(" = ", 1)
                current_stage[key.strip()] = _coerce_saved_value(val.strip())

    if current_stage is not None:
        stages.append(current_stage)

    return stages


def infer_saved_stage_index_from_step(saved_stages: List[Dict[str, Any]], step_count: int) -> int:
    """
    Infer which saved curriculum stage a checkpoint belongs to based on cumulative saved stage timesteps.
    """
    if len(saved_stages) == 0 or step_count < 0:
        return 0

    cumulative = 0
    for i, st in enumerate(saved_stages):
        stage_steps = int(st.get("timesteps", 0))
        cumulative += max(0, stage_steps)
        if step_count <= cumulative:
            return i

    return len(saved_stages) - 1


def apply_saved_stage_to_cfg_and_weights(
    cfg: CR3BPConfig,
    weights: RewardWeights,
    saved_stage: Dict[str, Any],
) -> Tuple[CR3BPConfig, RewardWeights]:
    """
    Apply one saved curriculum stage block onto cfg + reward weights.
    """
    cfg_out = CR3BPConfig(**vars(cfg))
    w_out = RewardWeights(**vars(weights))

    # Config fields that belong to CR3BPConfig and may vary by stage
    for field_name in cfg_out.__dataclass_fields__.keys():
        if field_name in saved_stage:
            setattr(cfg_out, field_name, saved_stage[field_name])

    # Reward weights
    for field_name in w_out.__dataclass_fields__.keys():
        if field_name in saved_stage:
            setattr(w_out, field_name, float(saved_stage[field_name]))

    return cfg_out, w_out


def infer_curriculum_stage_from_step(curriculum: List[CurriculumStage], step_count: int) -> int:
    if step_count < 0:
        return 0

    cumulative = 0
    for i, stage in enumerate(curriculum):
        cumulative += int(stage.timesteps)
        if step_count <= cumulative:
            return i

    return len(curriculum) - 1


def get_curriculum_resume_position(curriculum: List[CurriculumStage], total_steps: int):
    """
    Convert absolute trained step count into:
    - stage index
    - steps already consumed inside that stage
    - remaining steps inside that stage
    """
    total_steps = max(0, int(total_steps))

    cumulative_before = 0

    for i, stage in enumerate(curriculum):
        stage_steps = int(stage.timesteps)
        cumulative_after = cumulative_before + stage_steps

        if total_steps < cumulative_after:
            used_in_stage = total_steps - cumulative_before
            remaining_in_stage = stage_steps - used_in_stage
            return {
                "stage_idx": i,
                "stage_name": stage.name,
                "stage_total_steps": stage_steps,
                "used_in_stage": used_in_stage,
                "remaining_in_stage": remaining_in_stage,
                "cumulative_before_stage": cumulative_before,
            }

        cumulative_before = cumulative_after

    # If already beyond full curriculum, place at final stage with zero remaining
    last_idx = len(curriculum) - 1
    last_stage = curriculum[last_idx]
    return {
        "stage_idx": last_idx,
        "stage_name": last_stage.name,
        "stage_total_steps": int(last_stage.timesteps),
        "used_in_stage": int(last_stage.timesteps),
        "remaining_in_stage": 0,
        "cumulative_before_stage": cumulative_before - int(last_stage.timesteps),
    }


def build_cfg_and_weights_from_policy(policy_path: Path):
    run_dir = policy_path.parent
    saved = parse_run_config_txt(run_dir)

    cfg = CR3BPConfig()

    # Rebuild base cfg directly from saved run_config.txt
    for field_name in cfg.__dataclass_fields__.keys():
        if field_name in saved:
            setattr(cfg, field_name, saved[field_name])

    # Rebuild reward weights from saved run_config.txt fallback values
    weights = RewardWeights(
        w_flyby=float(saved.get("w_flyby", RewardWeights().w_flyby)),
        w_velocity=float(saved.get("w_velocity", RewardWeights().w_velocity)),
        w_dv=float(saved.get("w_dv", RewardWeights().w_dv)),
        w_return=float(saved.get("w_return", RewardWeights().w_return)),
        w_budget=float(saved.get("w_budget", RewardWeights().w_budget)),
        w_escape=float(saved.get("w_escape", RewardWeights().w_escape)),
        w_earth_crash=float(saved.get("w_earth_crash", RewardWeights().w_earth_crash)),
        w_moon_crash=float(saved.get("w_moon_crash", RewardWeights().w_moon_crash)),
        w_postflyby_earth_crash=float(
            saved.get("w_postflyby_earth_crash", RewardWeights().w_postflyby_earth_crash)
        ),
        w_invalid_preflyby_earth_return=float(
            saved.get(
                "w_invalid_preflyby_earth_return",
                RewardWeights().w_invalid_preflyby_earth_return,
            )
        ),
    )

    step_count = extract_step_from_policy_name(policy_path)

    saved_stages = parse_saved_curriculum_stages(run_dir)

    chosen_stage_idx = -1
    chosen_stage_name = "from_saved_run_config"

    if len(saved_stages) > 0:
        chosen_stage_idx = infer_saved_stage_index_from_step(saved_stages, step_count)
        chosen_stage = saved_stages[chosen_stage_idx]

        cfg, weights = apply_saved_stage_to_cfg_and_weights(cfg, weights, chosen_stage)
        chosen_stage_name = str(chosen_stage.get("stage_name", f"stage_{chosen_stage_idx+1}"))

    recovered = {
        "policy_step": step_count,
        "stage_idx": int(chosen_stage_idx),
        "stage_name": chosen_stage_name,
        "mcc_enabled": cfg.mcc_enabled,
        "tli_only_mode": cfg.tli_only_mode,
        "reward_after_tli_ballistic_enabled": cfg.reward_after_tli_ballistic_enabled,
        "config_file_found": (run_dir / "run_config.txt").exists(),
        "trainer_mode": getattr(cfg, "trainer_mode", "ppo_a"),
        "tli_control_mode": getattr(cfg, "tli_control_mode", "full"),
    }

    return cfg, weights, recovered





def load_and_rollout():
    
    from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2

    script_path = __file__
    saved_root = get_saved_root(script_path)
    policy_files = list_policy_files(saved_root)
    chosen = choose_from_list(policy_files, title="Select a saved policy (.zip)")

    cfg, weights, recovered = build_cfg_and_weights_from_policy(chosen)

    print("\nRecovered rollout config")
    print("-" * 60)
    print(f"policy file         : {chosen.name}")
    print(f"policy step         : {recovered['policy_step']}")
    print(f"trainer_mode        : {recovered['trainer_mode']}")
    print(f"tli_control_mode    : {recovered['tli_control_mode']}")
    print(f"inferred stage      : {recovered['stage_name']}")
    print(f"mcc_enabled         : {recovered['mcc_enabled']}")
    print(f"config file found   : {recovered['config_file_found']}")
    print(f"tli_only_mode       : {recovered['tli_only_mode']}")
    print(f"reward_after_tli    : {recovered['reward_after_tli_ballistic_enabled']}")
    print("-" * 60)

    env = CR3BPFreeReturnEnv(
        cfg,
        seed=RUN.eval_seed,
        reward_model=RewardFunction(RewardConfig(), weights),
    )
    env.set_debug_eval(True)

    model = TimeAwareRecurrentPPOv2.load(str(chosen), device=RUN.device)

    out_dir = ensure_dir(chosen.parent / f"rollout_{timestamp_str()}")
    rewards, terms_ts, info_last, reward_records, reward_audit = collect_episode_reward_timeseries(
        env,
        model,
        deterministic=True,
    )


    traj = np.array(env.traj, dtype=np.float64)
    t_hist = np.array(env.t_hist, dtype=np.float64)

    plot_trajectory(
        cfg,
        traj,
        burns=np.array(env.burns, dtype=np.float64) if len(env.burns) > 0 else None,
        burn_events=getattr(env, "burn_events", None),
        ballistic_ref_traj=getattr(env, "ballistic_ref_traj", None),
        ballistic_terminal_marker=getattr(env, "ballistic_terminal_marker_rot", None),
        terminal_marker=getattr(env, "terminal_marker_rot", None),
        title=f"Rotating (loaded) {chosen.stem}",
        out_path=str(out_dir / "traj_rot.png")
    )
    if bool(getattr(RUN, "generate_mcc_eval_plot", False)):
        plot_trajectory_mcc_debug(
            cfg,
            traj,
            burn_events=getattr(env, "burn_events", None),
            ballistic_ref_traj=getattr(env, "ballistic_ref_traj", None),
            ballistic_terminal_marker=getattr(env, "ballistic_terminal_marker_rot", None),
            terminal_marker=getattr(env, "terminal_marker_rot", None),
            mcc_ballistic_overlays=getattr(env, "mcc_ballistic_overlays", None),
            title=f"MCC debug (loaded) {chosen.stem}",
            out_path=str(out_dir / "traj_mcc_debug.png")
        )
    plot_trajectory_earth_centered_inertial(
        cfg,
        traj,
        t_hist,
        ballistic_ref_traj=getattr(env, "ballistic_ref_traj", None),
        ballistic_ref_t_hist=getattr(env, "ballistic_ref_t_hist", None),
        title=f"Inertial (loaded) {chosen.stem}",
        out_path=str(out_dir / "traj_inert.png")
    )


    report_path = save_episode_report_txt(
        out_dir=out_dir,
        stem="loaded_rollout",
        env=env,
        rewards=rewards,
        terms_ts=terms_ts,
        info_last=info_last,
        reward_records=reward_records,
        audit=reward_audit,
    )

    print("\nRollout finished.")
    print(f"success={info_last.get('success', False)} reason={info_last.get('term_reason','')}")
    print(f"dv_used={info_last.get('dv_used', np.nan)} min_rM={info_last.get('min_rM', np.nan)}")
    print(f"Plots saved in: {out_dir}")
    print(f"Episode txt report: {report_path}")



def batch_eval_saved_policy(n_eval_episodes: int = SAVED_POLICY_BATCH_EVAL_EPISODES):
    from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2

    script_path = __file__
    saved_root = get_saved_root(script_path)
    policy_files = list_policy_files(saved_root)
    chosen = choose_from_list(policy_files, title="Select a saved policy (.zip) for batch evaluation")

    cfg, weights, recovered = build_cfg_and_weights_from_policy(chosen)

    print("\nBatch saved-policy evaluation")
    print("-" * 60)
    print(f"policy file         : {chosen.name}")
    print(f"policy step         : {recovered['policy_step']}")
    print(f"stage used          : {recovered['stage_name']}")
    print(f"trainer_mode        : {recovered['trainer_mode']}")
    print(f"episodes            : {n_eval_episodes}")
    print("-" * 60)

    eval_env = CR3BPFreeReturnEnv(
        cfg,
        seed=RUN.eval_seed,
        reward_model=RewardFunction(RewardConfig(), weights),
    )
    eval_env.set_debug_eval(True)

    model = TimeAwareRecurrentPPOv2.load(str(chosen), device=RUN.device)

    out_dir = ensure_dir(chosen.parent / f"batch_eval_{timestamp_str()}")

    eval_results: List[Dict[str, Any]] = []
    reasons: Dict[str, int] = {}

    for _ in range(int(n_eval_episodes)):
        ep = run_eval_episode_collect(
            model=model,
            env=eval_env,
            deterministic=True,
            action_debug=False,
            capture_plot_data=True,
        )
        eval_results.append(ep)

        reason = str(ep["reason"])
        reasons[reason] = reasons.get(reason, 0) + 1

    ep_success = [1.0 if ep["ballistic_success"] else 0.0 for ep in eval_results]
    ep_trajectory_success = [1.0 if ep["trajectory_success"] else 0.0 for ep in eval_results]
    ep_success_flag = [1.0 if ep["success_flag_latched"] else 0.0 for ep in eval_results]

    ep_flyby = [1.0 if ep["flyby_done"] else 0.0 for ep in eval_results]
    ep_corridor = [1.0 if ep["corridor_hit"] else 0.0 for ep in eval_results]
    ep_rewards = [ep["reward_sum"] for ep in eval_results]
    ep_dv = [ep["dv_used"] for ep in eval_results]
    ep_min_rM = [ep["min_rM"] for ep in eval_results]
    ep_return_miss = [ep["return_corridor_miss"] for ep in eval_results]

    ep_preserved = [
        1.0 if (ep["ballistic_success"] and ep["trajectory_success"]) else 0.0
        for ep in eval_results
    ]
    ep_degraded = [
        1.0 if (ep["ballistic_success"] and (not ep["trajectory_success"])) else 0.0
        for ep in eval_results
    ]
    ep_rescued = [
        1.0 if ((not ep["ballistic_success"]) and ep["trajectory_success"]) else 0.0
        for ep in eval_results
    ]
    ep_unchanged_bad = [
        1.0 if ((not ep["ballistic_success"]) and (not ep["trajectory_success"])) else 0.0
        for ep in eval_results
    ]

    lines = []
    lines.append("=" * 90)
    lines.append("BATCH SAVED-POLICY EVALUATION")
    lines.append("=" * 90)
    lines.append(f"policy_file              : {chosen.name}")
    lines.append(f"policy_step              : {recovered['policy_step']}")
    lines.append(f"saved_stage_used         : {recovered['stage_name']}")
    lines.append(f"trainer_mode             : {recovered['trainer_mode']}")
    lines.append(f"episodes                 : {int(n_eval_episodes)}")
    lines.append("")
    lines.append(f"success_rate             : {safe_mean(ep_success):.6f}    (ballistic IC success)")
    lines.append(f"trajectory_success_rate  : {safe_mean(ep_trajectory_success):.6f}    (actual controlled trajectory)")
    lines.append(f"success_flag_rate        : {safe_mean(ep_success_flag):.6f}")
    lines.append(f"preservation_rate       : {safe_mean(ep_preserved):.6f}")
    lines.append(f"degradation_rate        : {safe_mean(ep_degraded):.6f}")
    lines.append(f"rescue_rate             : {safe_mean(ep_rescued):.6f}")
    lines.append(f"unchanged_bad_rate      : {safe_mean(ep_unchanged_bad):.6f}")
    lines.append(f"flyby_rate               : {safe_mean(ep_flyby):.6f}")
    lines.append(f"corridor_hit_rate        : {safe_mean(ep_corridor):.6f}")
    lines.append(f"mean_reward              : {safe_mean(ep_rewards):.6f}")
    lines.append(f"std_reward               : {safe_std(ep_rewards):.6f}")
    lines.append(f"mean_dv_used             : {safe_mean(ep_dv):.6f}")
    lines.append(f"mean_min_rM              : {safe_mean(ep_min_rM):.6f}")
    lines.append(f"mean_return_miss         : {safe_mean(ep_return_miss):.6f}")
    lines.append(f"termination_reasons      : {reasons}")
    lines.append("=" * 90)

    summary_txt = "\n".join(lines)

    with open(out_dir / "batch_eval_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    # Save one representative plotted episode
    success_eps = [ep for ep in eval_results if ep["success_strict"]]
    flyby_eps = [ep for ep in eval_results if ep["flyby_done"]]
    pool = success_eps if len(success_eps) > 0 else (flyby_eps if len(flyby_eps) > 0 else eval_results)
    plot_ep = pool[np.random.randint(len(pool))]

    traj = np.asarray(plot_ep["traj"], dtype=np.float64)
    t_hist = np.asarray(plot_ep["t_hist"], dtype=np.float64)
    rewards = np.asarray(plot_ep["rewards"], dtype=np.float64)
    terms_ts = plot_ep["terms_ts"]
    info_last = plot_ep["info_last"]


    plot_trajectory(
        cfg,
        traj,
        burns=plot_ep["burns"],
        burn_events=plot_ep["burn_events"],
        ballistic_ref_traj=plot_ep["ballistic_ref_traj"],
        ballistic_terminal_marker=plot_ep.get("ballistic_terminal_marker_rot", None),
        terminal_marker=plot_ep.get("terminal_marker_rot", None),
        title=f"Batch eval rotating example ({chosen.stem})",
        out_path=str(out_dir / "traj_rot_example.png")
    )
    if bool(getattr(RUN, "generate_mcc_eval_plot", False)):
        plot_trajectory_mcc_debug(
            cfg,
            traj,
            burn_events=plot_ep["burn_events"],
            ballistic_ref_traj=plot_ep["ballistic_ref_traj"],
            ballistic_terminal_marker=plot_ep.get("ballistic_terminal_marker_rot", None),
            terminal_marker=plot_ep.get("terminal_marker_rot", None),
            mcc_ballistic_overlays=plot_ep["mcc_ballistic_overlays"],
            title=f"Batch eval MCC debug example ({chosen.stem})",
            out_path=str(out_dir / "traj_mcc_debug_example.png")
        )

    plot_trajectory_earth_centered_inertial(
        cfg,
        traj,
        t_hist,
        ballistic_ref_traj=plot_ep["ballistic_ref_traj"],
        ballistic_ref_t_hist=plot_ep["ballistic_ref_t_hist"],
        title=f"Batch eval inertial example ({chosen.stem})",
        out_path=str(out_dir / "traj_inert_example.png")
    )


    report_path = save_episode_report_txt(
        out_dir=out_dir,
        stem="batch_eval_example",
        env=plot_ep["env_snapshot"],
        rewards=rewards,
        terms_ts=terms_ts,
        info_last=info_last,
    )

    print("\n" + summary_txt + "\n")
    print(f"Batch eval outputs saved in: {out_dir}")
    print(f"Example episode report     : {report_path}")


# ============================================================
# 7) MAIN
# ============================================================

def main():
    print("\n" + "=" * 78)
    print("CR3BP Free-Return RL  | tau: time-warp")
    
    
    print("=" * 78)
    print("[1] Train new policy PPO_TLI (PPO_A)")
    print("[2] Train new policy PPO_MCC (PPO_B)")
    print("[3] Load existing policy and run one eval rollout + plot")
    print("[4] Launch timelapse animator")
    print("[5] Continue training existing policy")
    print("[6] Launch manual override env")
    print("[7] Use pre selected checkpoint (Ubuntu/server)")
    print("[8] Batch evaluate saved policy")
    print("=" * 78)

    mode = input("Select mode (1/2/3/4/5/6/7/8): ").strip()

    if mode == "1":
        train(training_profile="ppo_tli")

    elif mode == "2":
        train(training_profile="ppo_mcc")

    elif mode == "3":
        load_and_rollout()

    elif mode == "4":
        launch_timelapse_animator()

    elif mode == "5":
        script_path = __file__
        saved_root = get_saved_root(script_path)
        policy_files = list_policy_files(saved_root)
        chosen = choose_from_list(policy_files, title="Select a saved policy (.zip) to continue training")

        print("\nSelect continuation curriculum:")
        print("[1] Continue with PPO_TLI (PPO_A) curriculum")
        print("[2] Continue with PPO_MCC (PPO_B) curriculum")
        submode = input("Select continuation type (1/2): ").strip()

        if submode == "1":
            train(training_profile="ppo_tli", resume_policy_path=chosen)
        elif submode == "2":
            train(training_profile="ppo_mcc", resume_policy_path=chosen)
        else:
            print("Invalid continuation type.")

    elif mode == "6":
        launch_manual_override_env()

    elif mode == "7":
        chosen = get_preselected_checkpoint_path()
        print("Using pre selected checkpoint for continuation training.")
        print(f"Checkpoint: {chosen}")

        print("\nSelect curriculum for this checkpoint:")
        print("[1] PPO_TLI (PPO_A)")
        print("[2] PPO_MCC (PPO_B)")
        submode = input("Select training type (1/2): ").strip()

        if submode == "1":
            train(training_profile="ppo_tli", resume_policy_path=chosen)
        elif submode == "2":
            train(training_profile="ppo_mcc", resume_policy_path=chosen)
        else:
            print("Invalid training type.")

    elif mode == "8":
        batch_eval_saved_policy()

    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()