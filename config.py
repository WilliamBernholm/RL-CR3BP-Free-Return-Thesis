"""
============================================================
CENTRAL CONFIGURATION LAYER FOR THE CR3BP RL PROJECT
============================================================

This module defines the shared configuration objects used across
the Earth-Moon CR3BP reinforcement learning project.

It acts as the central experiment-definition layer for:
- training settings
- reward shaping settings
- curriculum stage settings
- base environment settings
- helper utilities for applying overrides

------------------------------------------------------------
WHAT THIS SCRIPT CONTAINS
------------------------------------------------------------

This script defines:

1) RunConfig
- global run and training settings
- PPO hyperparameters
- rollout / evaluation settings
- drift-time limits and propagation-control settings
- user-facing burn-cap and diagnostic settings

2) RewardConfig
- reward-shaping parameters used by the reward function
- flyby shaping scales
- return-corridor shaping scales
- velocity / delta-v shaping settings

3) RewardWeights
- weighting factors for the main reward terms
- used to change mission priorities by curriculum stage

4) CurriculumStage
- stage-specific training settings
- mission mode (PPO-A or PPO-B)
- staged-TLI controls
- PPO-B scenario-library controls
- per-stage execution noise
- per-stage reward weights and entropy settings

5) CR3BPConfig
- base environment configuration used by the CR3BP env
- physical thresholds and mission radii
- observation-space toggles
- trainer-mode settings
- PPO-B handoff / library settings
- staged-TLI settings

6) Global helpers
- RUN = RunConfig()
- apply_overrides(...)
- ppo_rollout_block_size()

------------------------------------------------------------
PROJECT ROLE
------------------------------------------------------------

This module does not run training, propagation, plotting, or reward
evaluation directly.

Instead, it defines the parameter structures that are shared by:
- train_ppo_v4.py
- cr3bp_env_v4.py
- curriculum_ppoa.py
- curriculum_ppob.py
- plotting / reporting tools

It is intended to be the single place where the projects default
experiment settings are declared and documented.

------------------------------------------------------------
HOW CONFIGURATION FLOWS THROUGH THE PROJECT
------------------------------------------------------------

The intended flow is:

- config.py provides default shared dataclasses
- curriculum files define stage lists and optional overrides
- the training script selects a profile and curriculum
- overrides are applied onto the default config objects
- the environment receives a stage-specific CR3BPConfig
- the reward function receives the selected RewardConfig and
  RewardWeights

This allows PPO-A and PPO-B to share one clean configuration
framework while still using different curricula and overrides.

------------------------------------------------------------
SUPPORTED TRAINING CONTEXT
------------------------------------------------------------

PPO-A:
- used for TLI optimization from LEO
- typically starts from a spacecraft state in circular Earth orbit
- may use staged-TLI curriculum settings and TLI-specific reward shaping

PPO-B:
- used for MCC optimization from known post-TLI handoff states
- typically starts from saved scenario-library states
- may use PPO-B-specific reward and run overrides

------------------------------------------------------------
CHANGES RELATIVE TO A SIMPLE BASELINE CONFIG SCRIPT
------------------------------------------------------------

(Changed from baseline: supports both PPO-A and PPO-B from one shared
configuration layer.)

(Changed from baseline: separates global defaults from curriculum-stage
overrides.)

(Changed from baseline: supports curriculum-controlled PPO-B scenario
library settings and staged-TLI controls.)

(Changed from baseline: supports override application at runtime through
apply_overrides(...).)

============================================================
"""




from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 0) TOP CONFIG YOU EDIT
# ============================================================

@dataclass
class RunConfig:
    # ---------- workflow ----------
    enable_plotting: bool = True
    saved_root_name: str = "Saved Policies"

    # ---------- training ----------
    total_timesteps: int = 2_000_000 #unused
    n_envs: int = 8
    train_seed: int = 1000
    eval_seed: int = 999

    eval_interval_steps: int = 2_048*2
    eval_episodes: int = 16
    plot_every_evals: int = 1

    checkpoint_every: int =2_048*20

    # ---------- PPO-LSTM ----------
    gamma: float = 0.997
    gae_lambda: float = 0.95
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 8
    learning_rate: float = 0.0001
    clip_range: float = 0.15
    max_grad_norm: float = 0.5
    ent_coef_default: float = 0.003
    device: str = "auto"


    # ======================================================
    # STATE-DEPENDENT TAU MASKING
    # PRE_TLI  : fine phasing in LEO
    # POST_TLI : larger coasting after committed departure
    # ======================================================
    drift_min_minutes_pre_tli: float = 5.0 / 60.0
    drift_max_minutes_pre_tli: float = 1.0 

    drift_min_minutes_post_tli: float = 10.0
    drift_max_minutes_post_tli: float = 3000.0 

    # tiny PRE_TLI burns are masked to zero to stop LEO dithering
    pre_tli_burn_deadzone_frac_of_tli_cap: float = 0.0

    
    # reference circular LEO orbits
    no_tli_terminate_after_leo_orbits: float = 3.0

    # ======================================================
    # V3_2 PROPAGATION TARGET SUBSTEP POLICY
    # We do NOT want a fixed 1-minute RK4 target for very long coasts,
    # because that would be too slow.
    #
    # Instead:
    # - short drifts use ~1 minute RK4 steps
    # - long drifts are allowed to use coarser RK4 steps
    # ======================================================
    rk4_substep_target_min_minutes: float = 1.0
    rk4_substep_target_max_minutes: float = 5.0

    # Drift duration at which we transition from min to max RK4 target step.
    # Example:
    #   dt_total <= 10 min   -> target ~1 min
    #   dt_total >= 300 min  -> target ~5 min
    rk4_target_transition_min_minutes: float = 10.0
    rk4_target_transition_max_minutes: float = 300.0


    # ======================================================
    # V3_2 REGION-BASED INTEGRATION REFINEMENT
    # Use finer RK4 target steps near Earth / Moon.
    # ======================================================
    fine_substep_region_radius: float = 0.1

    # Inside the fine region, force the RK4 target substep to this value.
    fine_rk4_substep_minutes: float = 1.0


    # Scale factor for immediate ballistic reward after TLI.
    # 0.1 means keep only 10% of that reward.
    tli_ballistic_scale: float = 0.7

    # ======================================================
    # FIXED GLOBAL TLI DV MAPPING
    # ======================================================
    # IMPORTANT:
    # Same latent action should always mean the same physical DV.
    # Therefore:
    # - tli_dv_min / tli_dv_max define ONE GLOBAL PHYSICAL RANGE
    # - curriculum only restricts allowed latent u interval
    #
    # If None:
    #   min defaults to 0.0
    #   max defaults to dv_cap_tli
    tli_dv_min: Optional[float] = None
    tli_dv_max: Optional[float] = None

   
    # ---------- authority ----------
    use_single_dv_cap: bool = False
    dv_cap_single: float = 4.4

    # ---------- user-facing DV caps in km/s ----------
    # If set, these override cfg.dv_max_tli / cfg.dv_max_mcc after conversion to nondim.
    tli_dv_max_kms: Optional[float] = 0.40
    mcc_dv_max_kms: Optional[float] = 0.03

    # Earth-Moon CR3BP characteristic scales for km/s -> nondim conversion
    cr3bp_Lstar_km: float = 384400.0
    cr3bp_Tstar_s: float = 375200.0



    use_global_burn_cap_kms: bool = False
    global_burn_cap_kms: float = 3.5


    # TLI ballistic reward trigger:
    # the first time either condition is met, we declare that the real TLI
    # has happened and evaluate the ballistic reward once.
    #
    # 1) burn magnitude threshold in km/s
    # 2) departure radius threshold from Earth in nondim units
    tli_ballistic_trigger_kms: float = 3.1
    tli_departure_trigger_rE: float = 0.1

    plot_spawn_sweep_enabled: bool = False
    plot_spawn_sweep_every_evals: int = 1
    plot_spawn_sweep_count: int = 8
    plot_spawn_sweep_deterministic: bool = True

        
    # ---------- MCC ballistic overlay diagnostics ----------
    # If True, eval/debug episodes will build a ballistic coast-only branch
    # immediately after each real MCC burn.
    generate_mcc_eval_plot: bool = True

    # Ignore very tiny MCC burns when generating overlay branches.
    # This threshold is in km/s.
    mcc_overlay_min_dv_kms: float = 0.0005


# ============================================================
# 1) SEAN-STYLE REWARD
# ============================================================

@dataclass
class RewardConfig:
    v_target_moon: float = 0.91067
    v_deadzone: float = 0.05

    # Flyby shaping: broader, more generous signal
    beta_distance_flyby: float = 2.2
    r0_distance_flyby: float = 0.18

    # Return corridor shaping: keep old behavior
    beta_distance_return: float = 3.5
    r0_distance_return: float = 0.1

    dv_budget: float = 15.0
    dv_scale: float = 1.0

    earth_radius: float = 0.014
    moon_radius: float = 0.005

    flyby_reward_gate: float = 1.10


@dataclass
class RewardWeights:
    w_flyby: float = 90.0
    w_velocity: float = 0.0
    w_dv: float = 2.0
    w_return: float = 90.0

    w_budget: float = 500.0
    w_escape: float = 90.0

    w_earth_crash: float = 500.0
    w_moon_crash: float = 500.0
    w_postflyby_earth_crash: float = 30.0

    # New: terminal penalty for a post-TLI, pre-flyby ballistic branch
    # that turns back toward Earth before any lunar flyby
    w_invalid_preflyby_earth_return: float = 60.0


@dataclass
class CurriculumStage:
    name: str
    reward_weights: RewardWeights
    entropy_coef: float
    timesteps: int

    # ---------- trainer / mission mode ----------
    # "ppo_a"                  -> normal LEO spawn, train TLI
    # "ppo_b_baseline"         -> spawn from user-defined post-TLI baseline
    # "ppo_b_from_external_ic" -> spawn from externally supplied post-TLI IC
    trainer_mode: str = "ppo_a"



    # ---------- staged TLI experiment ----------
    staged_tli_enabled: bool = False
    staged_tli_commit_on_cumulative_dv: bool = True
    staged_tli_limit_burn_count: bool = True
    staged_tli_max_burn_count: int = 40
    staged_tli_min_commit_frac_of_target: float = 1.0
    staged_tli_cumulative_dv_target: Optional[float] = None


    # ---------- PPO-B scenario library mode ----------
    # "baseline"         -> existing baseline builder
    # "external_ic"      -> existing external payload mode
    # "scenario_library" -> sample a nominal TLI seed from a saved .npz file
    ppo_b_case_source: str = "scenario_library"

    # path to .npz scenario library
    ppo_b_library_path: str = "rough_scenario_classification/ppob_handoff_states_30min.npz"

    # class probabilities for scenario library sampling
    ppo_b_prob_good: float = 0
    ppo_b_prob_savable: float = 1
    ppo_b_prob_bad: float = 0

    # evaluation uses same distribution/noise unless you later override externally
    ppo_b_eval_use_same_distribution: bool = True

    # physical seed noise BEFORE building post-TLI state
    # theta noise in degrees
    ppo_b_noise_theta_deg: float = 0.0

    # rotate the nominal TLI burn direction by this Gaussian sigma in degrees
    ppo_b_noise_tli_dir_deg: float = 0.0

    # add Gaussian sigma to the nominal TLI magnitude in km/s
    ppo_b_noise_tli_dv_kms: float = 0.0


    # ---------- fixed single-case training mode ----------
    # If True, ignore label-probability sampling and always use one exact library index
    ppo_b_use_fixed_index: bool = False

    # Exact row inside the loaded PPO-B library
    ppo_b_fixed_index: int = 0

    # Optional local state noise applied AFTER loading the chosen handoff state
    # Only used for handoff-state libraries
    ppo_b_fixed_state_noise_pos: float = 0.0
    ppo_b_fixed_state_noise_vel: float = 0.0

    # ---------- TLI control mode for PPO-A ----------
    # "full"        -> current behavior, xy burn vector
    # "tangential"  -> pre-TLI burn projected to local tangential direction
    tli_control_mode: str = "full"

    # ---------- mission structure ----------
    mcc_enabled: bool = True
    tli_only_mode: bool = False
    reward_after_tli_ballistic_enabled: bool = False

    # ---------- spawn curriculum ----------
    spawn_theta_limit_enabled: bool = True
    spawn_theta_min: float = 4.04056
    spawn_theta_max: float = 4.04056

    # ---------- PPO-B baseline initial condition ----------
    # These define a baseline TLI applied from a chosen LEO spawn. 
    # based on "eval0018_step000036864_episode_report.txt" of 
    # PPOA__MILESTONE_BEST_SINGLE_BALLISTIC_210.000_STEP_36864__2026-04-11_20-11-04.zip
    ppo_b_baseline_theta: float = 4.7181
    ppo_b_baseline_ax: float = 0.9751105905
    ppo_b_baseline_ay: float = -0.4365218282
    ppo_b_baseline_tau: float = -0.199569

    # additional optional noise applied AFTER the baseline TLI+coast spawn
    ppo_b_baseline_state_noise_pos: float = 0
    ppo_b_baseline_state_noise_vel: float = 0

    # ---------- execution noise ----------
    dv_noise_sigma_tli: float = 0.0
    dv_noise_sigma_mcc: float = 0.0

    # ---------- policy std override ----------
    use_manual_log_std: bool = False
    manual_log_std_value: float = 0.0



@dataclass
class CR3BPConfig:
    mu: float = 0.012150585609624

    dt: float = 0.0048
    t_max: float = 2.4
    integration_substeps: int = 50

    r0_earth: float = 0.0176
    v_circ_earth: float = 7.5

    r_moon_flyby: float = 0.06
    r_earth_return: float = 0.05

    r_earth_impact: float = 0.014
    r_moon_impact: float = 0.0045

    r_escape: float = 2.0

    rp_min: float = 0.0143 #0.0174
    rp_max: float = 0.06

    dv_max_tli: float = 4.4
    dv_max_mcc: float = 0.1

    # These are stage-overridable execution noise levels
    dv_noise_sigma_tli: float = 0.0
    dv_noise_sigma_mcc: float = 0.0

    mcc_enabled: bool = True

    store_dense_training_traj: bool = False

    # observation scaling
    pos_scale: float = 1.0
    vel_scale: float = 10.0
    c_scale: float = 55.0

    add_phase_angle_obs: bool = True

    # If True, include the 4 legacy mixed TLI/MCC mode fields:
    # [tli_used_flag, tau_max_current_norm, dv_cap_current_norm, pre_tli_clock_norm]
    # Keep this toggleable so we can revert quickly if needed.
    add_legacy_mode_obs: bool = False

    # ======================================================
    # MODE / MASK OBSERVATION
    # ======================================================
    add_mode_obs: bool = True

    # if True, terminate when spacecraft still has not reached the configured
    # departure / left-LEO radius after RUN.no_tli_terminate_after_leo_orbits
    terminate_if_no_leo_exit: bool = True

    # Consistent "left LEO / departure" radius.
    # Recommended: same value as RUN.tli_departure_trigger_rE
    left_leo_trigger_rE: float = 0.1

    # If the spacecraft has crossed left_leo_trigger_rE but still has not
    # achieved a real TLI, terminate after this extra grace time.
    left_leo_no_tli_grace_minutes: float = 60.0

    # ======================================================
    # V3 STAGE-APPLIED OPTIONS
    # ======================================================
    tli_only_mode: bool = False
    reward_after_tli_ballistic_enabled: bool = False

    spawn_theta_limit_enabled: bool = False
    spawn_theta_min: float = 0.0
    spawn_theta_max: float = 2.0 * math.pi



    terminate_on_dv_budget_exceed: bool = True

    # ------------------------------------------------------
    # Ballistic invalid pre-flyby Earth-return detection
    # Active only in ballistic TLI reward evaluation
    # ------------------------------------------------------
    ballistic_invalid_preflyby_return_enabled: bool = True

    # If the ballistic path never gets beyond this Earth radius,
    # treat it as an obvious Earth-bound invalid orbit.
    ballistic_invalid_stuck_max_rE: float = 0.15

    # Once the trajectory has exceeded this Earth radius,
    # we allow the "falling back to Earth" invalid test to arm.
    ballistic_invalid_return_arm_rE: float = 0.15

    # Trigger the fall-back invalid condition only if the
    # inward Earth radial speed is clearly negative.
    ballistic_invalid_return_vrE_threshold: float = -5e-3

    # Do NOT declare invalid if the spacecraft is already getting
    # meaningfully near the Moon.
    ballistic_invalid_return_moon_far_rM: float = 0.40

    # If a ballistic branch never exceeds this larger Earth radius,
    # we still consider it too low-energy / too Earth-bound to count
    # as a meaningful translunar attempt.
    ballistic_invalid_min_meaningful_outbound_rE: float = 0.70

    # ======================================================
    # V4 TRAINER MODE OPTIONS
    # ======================================================
    trainer_mode: str = "ppo_a"
    tli_control_mode: str = "full"

    # PPO-B baseline mode:
    # build a post-TLI initial condition from a hand-picked LEO spawn
    # and one baseline TLI command
    ppo_b_baseline_theta: float = 1.5 * math.pi
    ppo_b_baseline_ax: float = 1.0
    ppo_b_baseline_ay: float = 0.0
    ppo_b_baseline_tau: float = 0.0

    # Optional state perturbation after baseline post-TLI spawn
    ppo_b_baseline_state_noise_pos: float = 0.0
    ppo_b_baseline_state_noise_vel: float = 0.0


    # PPO-B scenario library mode
    ppo_b_case_source: str = "baseline"
    ppo_b_library_path: str = ""

    ppo_b_prob_good: float = 0.2
    ppo_b_prob_savable: float = 0.6
    ppo_b_prob_bad: float = 0.2

    ppo_b_eval_use_same_distribution: bool = True

    # physical seed noise before post-TLI state is built
    ppo_b_noise_theta_deg: float = 0.0
    ppo_b_noise_tli_dir_deg: float = 0.0
    ppo_b_noise_tli_dv_kms: float = 0.0

    # Fixed single-case PPO-B library mode
    ppo_b_use_fixed_index: bool = False
    ppo_b_fixed_index: int = 0
    ppo_b_fixed_state_noise_pos: float = 0.0
    ppo_b_fixed_state_noise_vel: float = 0.0


    # ======================================================
    # SWITCHABLE STAGED PRE-TLI MODE FOR PPO-A TESTING
    # ======================================================
    staged_tli_enabled: bool = False

    # If True, real TLI can commit once cumulative pre-TLI dv reaches target.
    staged_tli_commit_on_cumulative_dv: bool = True

    # cumulative pre-TLI dv target in nondim units
    staged_tli_cumulative_dv_target: float = 0.0

    # optional hard cap on number of pre-TLI burn steps
    staged_tli_limit_burn_count: bool = True
    staged_tli_max_burn_count: int = 12

    # if max burn count is reached, allow commit only if this minimum cumulative dv is met
    staged_tli_min_commit_frac_of_target: float = 0.85

    # add 2 extra mode obs:
    # [pre_tli_cum_dv_norm, pre_tli_burn_count_norm]
    add_staged_tli_obs: bool = True


RUN = RunConfig()



def apply_overrides(obj, overrides: dict | None):
    """
    Apply a dict of attribute overrides onto a dataclass-like config object.
    Unknown keys raise an error so mistakes are caught early.
    """
    if overrides is None:
        return obj

    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise AttributeError(
                f"{type(obj).__name__} has no attribute '{key}'"
            )
        setattr(obj, key, value)

    return obj



def ppo_rollout_block_size() -> int:
    return int(RUN.n_steps * RUN.n_envs)