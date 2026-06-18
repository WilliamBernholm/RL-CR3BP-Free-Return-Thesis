"""
============================================================
PPO-A CURRICULUM DEFINITION (TLI OPTIMIZATION FROM LEO)
============================================================

This script defines the training curriculum used for PPO-A.

PPO-A is responsible for learning the Trans-Lunar Injection (TLI)
phase, starting from a spacecraft in circular Low Earth Orbit (LEO).

------------------------------------------------------------
WHAT THIS SCRIPT DOES
------------------------------------------------------------

- Builds a list of CurriculumStage objects for PPO-A training
- Each stage defines:
  • reward weighting
  • entropy level (exploration vs exploitation)
  • training duration
  • staged-TLI behavior
  • execution noise levels

- Returns:
    (curriculum, overrides)

------------------------------------------------------------
MISSION CONTEXT (PPO-A)
------------------------------------------------------------

- Initial condition:
  Spacecraft starts in circular Earth orbit (LEO)

- Task:
  Learn how to perform TLI using:
    • burn direction
    • burn magnitude
    • phasing time (tau)

- After TLI:
  A ballistic trajectory is evaluated for flyby quality

------------------------------------------------------------
KEY DESIGN FEATURES
------------------------------------------------------------

STAGED TLI (main idea)
- TLI is not a single impulsive burn
- Agent builds TLI through multiple small burns
- Commit occurs when cumulative Δv reaches target

Ballistic reward after TLI
- Once TLI is detected, a one-shot ballistic evaluation is performed
- Encourages correct outbound geometry early

Low → high precision curriculum
- Stage 1: high entropy, tiny noise (exploration)
- Stage 2: reduced entropy, small noise
- Stage 3: low entropy, refinement

------------------------------------------------------------
OVERRIDES SYSTEM
------------------------------------------------------------

This script also defines overrides applied at runtime:

RUN overrides:
- generate_mcc_eval_plot = False
  (PPO-A does not use MCC diagnostics)

ENV overrides:
- add_staged_tli_obs = True
  (agent observes cumulative dv and burn count)

REWARD overrides:
- PPO-A-specific flyby shaping parameters

------------------------------------------------------------
CHANGES RELATIVE TO A SIMPLE RL BASELINE
------------------------------------------------------------

(Changed from baseline: introduces staged TLI instead of single-burn action)

(Changed from baseline: includes ballistic reward evaluation immediately after TLI)

(Changed from baseline: uses curriculum stages with decreasing entropy)

(Changed from baseline: includes structured override system for run/env/reward configs)

============================================================
"""

from __future__ import annotations

from config import CurriculumStage, RewardWeights


def build_curriculum_ppoa(kms_to_nondim_dv):
    MANUAL_AB_LIB = "rough_scenario_classification/manual_cases/ppob_case94_ab_library.npz"

    curriculum = [
        CurriculumStage(
            name="ppo_a_stage_1_staged_tli_tiny_noise",
            reward_weights=RewardWeights(
                w_flyby=40.0,
                w_velocity=0.0,
                w_dv=0.2,
                w_return=160.0,
                w_budget=80.0,
                w_escape=12.0,
                w_earth_crash=150.0,
                w_moon_crash=5.0,
                w_postflyby_earth_crash=90.0,
                w_invalid_preflyby_earth_return=10.0,
            ),
            entropy_coef=0.005,
            timesteps=400_000,
            trainer_mode="ppo_a",
            tli_control_mode="full",
            mcc_enabled=True,
            tli_only_mode=True,
            reward_after_tli_ballistic_enabled=True,
            staged_tli_enabled=True,
            staged_tli_commit_on_cumulative_dv=True,
            staged_tli_limit_burn_count=True,
            staged_tli_max_burn_count=60,
            staged_tli_min_commit_frac_of_target=1.0,
            staged_tli_cumulative_dv_target=kms_to_nondim_dv(3.1),
            spawn_theta_limit_enabled=True,
            spawn_theta_min=4.04056,
            spawn_theta_max=4.04056,
            ppo_b_case_source="scenario_library",
            ppo_b_library_path=MANUAL_AB_LIB,
            ppo_b_prob_good=0.0,
            ppo_b_prob_savable=0.5,
            ppo_b_prob_bad=0.5,
            ppo_b_eval_use_same_distribution=True,
            ppo_b_noise_theta_deg=0.0,
            ppo_b_noise_tli_dir_deg=0.0,
            ppo_b_noise_tli_dv_kms=0.0,
            ppo_b_use_fixed_index=False,
            ppo_b_fixed_index=0,
            ppo_b_fixed_state_noise_pos=0.0,
            ppo_b_fixed_state_noise_vel=0.0,
            dv_noise_sigma_tli=0.0,
            dv_noise_sigma_mcc=0.0,
            use_manual_log_std=False,
            manual_log_std_value=0.0,
        ),
        CurriculumStage(
            name="ppo_a_stage_2_staged_tli_small_noise",
            reward_weights=RewardWeights(
                w_flyby=40.0,
                w_velocity=0.0,
                w_dv=0.2,
                w_return=160.0,
                w_budget=80.0,
                w_escape=12.0,
                w_earth_crash=150.0,
                w_moon_crash=5.0,
                w_postflyby_earth_crash=100.0,
                w_invalid_preflyby_earth_return=10.0,
            ),
            entropy_coef=0.004,
            timesteps=200_000,
            trainer_mode="ppo_a",
            tli_control_mode="full",
            mcc_enabled=True,
            tli_only_mode=True,
            reward_after_tli_ballistic_enabled=True,
            staged_tli_enabled=True,
            staged_tli_commit_on_cumulative_dv=True,
            staged_tli_limit_burn_count=True,
            staged_tli_max_burn_count=60,
            staged_tli_min_commit_frac_of_target=1.0,
            staged_tli_cumulative_dv_target=kms_to_nondim_dv(3.1),
            spawn_theta_limit_enabled=True,
            spawn_theta_min=4.04056,
            spawn_theta_max=4.04056,
            ppo_b_case_source="scenario_library",
            ppo_b_library_path=MANUAL_AB_LIB,
            ppo_b_prob_good=0.0,
            ppo_b_prob_savable=0.5,
            ppo_b_prob_bad=0.5,
            ppo_b_eval_use_same_distribution=True,
            ppo_b_noise_theta_deg=0.0,
            ppo_b_noise_tli_dir_deg=0.0,
            ppo_b_noise_tli_dv_kms=0.0,
            ppo_b_use_fixed_index=False,
            ppo_b_fixed_index=0,
            ppo_b_fixed_state_noise_pos=0.0,
            ppo_b_fixed_state_noise_vel=0.0,
            dv_noise_sigma_tli=0.0,
            dv_noise_sigma_mcc=0.0,
            use_manual_log_std=False,
            manual_log_std_value=0.0,
        ),
        CurriculumStage(
            name="ppo_a_stage_3_staged_tli_refinment",
            reward_weights=RewardWeights(
                w_flyby=40.0,
                w_velocity=0.0,
                w_dv=0.2,
                w_return=160.0,
                w_budget=80.0,
                w_escape=12.0,
                w_earth_crash=150.0,
                w_moon_crash=5.0,
                w_postflyby_earth_crash=140.0,
                w_invalid_preflyby_earth_return=25.0,
            ),
            entropy_coef=0.002,
            timesteps=200_000,
            trainer_mode="ppo_a",
            tli_control_mode="full",
            mcc_enabled=True,
            tli_only_mode=True,
            reward_after_tli_ballistic_enabled=True,
            staged_tli_enabled=True,
            staged_tli_commit_on_cumulative_dv=True,
            staged_tli_limit_burn_count=True,
            staged_tli_max_burn_count=60,
            staged_tli_min_commit_frac_of_target=1.0,
            staged_tli_cumulative_dv_target=kms_to_nondim_dv(3.1),
            spawn_theta_limit_enabled=True,
            spawn_theta_min=4.04056,
            spawn_theta_max=4.04056,
            ppo_b_case_source="scenario_library",
            ppo_b_library_path=MANUAL_AB_LIB,
            ppo_b_prob_good=0.0,
            ppo_b_prob_savable=0.5,
            ppo_b_prob_bad=0.5,
            ppo_b_eval_use_same_distribution=True,
            ppo_b_noise_theta_deg=0.0,
            ppo_b_noise_tli_dir_deg=0.0,
            ppo_b_noise_tli_dv_kms=0.0,
            ppo_b_use_fixed_index=False,
            ppo_b_fixed_index=0,
            ppo_b_fixed_state_noise_pos=0.0,
            ppo_b_fixed_state_noise_vel=0.0,
            dv_noise_sigma_tli=0.0,
            dv_noise_sigma_mcc=0.0,
            use_manual_log_std=False,
            manual_log_std_value=0.0,
        ),
    ]

    overrides = {
        "run": {
            "generate_mcc_eval_plot": False,
        },
        "env": {
            "add_staged_tli_obs": True,
        },
        "reward": {
            # PPO-A-specific shaping
            "beta_distance_flyby": 2.2,
            "r0_distance_flyby": 0.18,
            
            "beta_distance_return": 3.5,
            "r0_distance_return": 0.1,
        },
    }

    return curriculum, overrides