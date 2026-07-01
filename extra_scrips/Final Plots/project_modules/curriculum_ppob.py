"""
============================================================
PPO-B CURRICULUM DEFINITION (MCC OPTIMIZATION FROM HANDOFF STATE)
============================================================

This script defines the training curriculum used for PPO-B.

PPO-B is responsible for learning Mid-Course Corrections (MCC)
starting from a known post-TLI handoff state.

------------------------------------------------------------
WHAT THIS SCRIPT DOES
------------------------------------------------------------

- Builds a list of CurriculumStage objects for PPO-B training
- Each stage defines:
  • reward weighting
  • entropy level
  • training duration
  • scenario-library sampling behavior
  • state noise levels

- Returns:
    (curriculum, overrides)

------------------------------------------------------------
MISSION CONTEXT (PPO-B)
------------------------------------------------------------

- Initial condition:
  Spacecraft starts from a saved post-TLI state
  loaded from a scenario library (.npz file)

- Task:
  Apply a single MCC burn to:
    • improve lunar flyby
    • reduce return error
    • avoid invalid Earth-return trajectories

------------------------------------------------------------
KEY DESIGN FEATURES
------------------------------------------------------------

Scenario-library initialization
- Instead of LEO, agent starts from realistic TLI outcomes
- Loaded from precomputed dataset

Single-case curriculum (current setup)
- Fixed scenario index used for all episodes
- Allows precise refinement before generalization

Progressive noise injection
- Stage 1: no noise (learn baseline correction)
- Stage 2: small state noise
- Stage 3: larger state noise + lower entropy

Reward emphasis shift
- Strong penalty on Δv usage
- Strong shaping for flyby and return geometry

------------------------------------------------------------
OVERRIDES SYSTEM
------------------------------------------------------------

RUN overrides:
- generate_mcc_eval_plot = True
  (PPO-B uses MCC ballistic overlay diagnostics)

ENV overrides:
- staged-TLI observations disabled
  (not relevant for MCC phase)

REWARD overrides:
- PPO-B-specific shaping tuned for post-TLI correction

------------------------------------------------------------
CHANGES RELATIVE TO A SIMPLE RL BASELINE
------------------------------------------------------------

(Changed from baseline: uses scenario-library states instead of random initialization)

(Changed from baseline: isolates MCC as a separate learning problem)

(Changed from baseline: uses fixed-case training before generalization)

(Changed from baseline: includes explicit override control for PPO-B diagnostics)

============================================================
"""

from __future__ import annotations

from config import CurriculumStage, RewardWeights
#"rough_scenario_classification/ppob_handoff_states_30min.npz"
#"rough_scenario_classification/test1.npz"
#65


# PPO-B dv normalization reference
# Hard-coded to current 30 m/s MCC authority.
# Do not automatically couple this to RUN.mcc_dv_max_kms.

#w_dv old baseline
# Additional reference points:
#
#   old 200  -> new ~6
#   old 210  -> new ~6.1
#   old 220  -> new ~6.4
PPOB_DV_REF_ND = 0.03 / (384400.0 / 375200.0)



def build_curriculum_ppob():

    from pathlib import Path

    # Regular baseline trejectory 
    #"rough_scenario_classification/ppob_handoff_states_30min.npz"
    #Intex 65

    MAIN_LIB = str(
        Path("rough_scenario_classification")
        / "Model_stage03_step00798720_R41.38_SR1.000_LD1.00455_CMnan_2026-05-23_18-51-29_staged_handoff_30min.npz"
    )
    
    MAIN_CASE_IDX = 0

    curriculum = [
        CurriculumStage(
            name="ppo_b_Shaping",
            reward_weights=RewardWeights(
                w_flyby=10.0,
                w_velocity=0.0,
                w_dv=12.0,
                w_return=100.0,
                w_budget=40.0,
                w_escape=60.0,
                w_earth_crash=150.0,
                w_moon_crash=20.0,
                w_postflyby_earth_crash=50.0,
                w_invalid_preflyby_earth_return=60.0,
            ),
            entropy_coef=0.009,
            timesteps=400_704,
            trainer_mode="ppo_b_library",
            tli_control_mode="full",
            mcc_enabled=True,
            tli_only_mode=False,
            reward_after_tli_ballistic_enabled=False,
            spawn_theta_limit_enabled=False,
            spawn_theta_min=0.0,
            spawn_theta_max=6.283185307179586,
            ppo_b_baseline_theta=4.7181,
            ppo_b_baseline_ax=0.9751105905,
            ppo_b_baseline_ay=-0.4365218282,
            ppo_b_baseline_tau=-0.199569,
            ppo_b_baseline_state_noise_pos=0.0,
            ppo_b_baseline_state_noise_vel=0.0,
            ppo_b_case_source="scenario_library",
            ppo_b_library_path=MAIN_LIB,
            ppo_b_prob_good=0.0,
            ppo_b_prob_savable=1.0,
            ppo_b_prob_bad=0.0,
            ppo_b_eval_use_same_distribution=True,
            ppo_b_noise_theta_deg=0.0,
            ppo_b_noise_tli_dir_deg=0.0,
            ppo_b_noise_tli_dv_kms=0.0,
            ppo_b_use_fixed_index=True,
            ppo_b_fixed_index=MAIN_CASE_IDX,
            ppo_b_fixed_state_noise_pos=0.0,
            ppo_b_fixed_state_noise_vel=0.0,
            dv_noise_sigma_tli=0.0,
            dv_noise_sigma_mcc=0.0,
            use_manual_log_std=False,
            manual_log_std_value=0.0,
        ),
        CurriculumStage(
            name="ppo_b_Dv_refine_",
            reward_weights=RewardWeights(
                w_flyby=10.0,
                w_velocity=0.0,
                w_dv=13.0,
                w_return=100.0,
                w_budget=40.0,
                w_escape=60.0,
                w_earth_crash=150.0,
                w_moon_crash=20.0,
                w_postflyby_earth_crash=50.0,
                w_invalid_preflyby_earth_return=60.0,
            ),
            entropy_coef=0.006,
            timesteps=100_000,
            trainer_mode="ppo_b_library",
            tli_control_mode="full",
            mcc_enabled=True,
            tli_only_mode=False,
            reward_after_tli_ballistic_enabled=False,
            spawn_theta_limit_enabled=False,
            spawn_theta_min=0.0,
            spawn_theta_max=6.283185307179586,
            ppo_b_baseline_theta=4.7181,
            ppo_b_baseline_ax=0.9751105905,
            ppo_b_baseline_ay=-0.4365218282,
            ppo_b_baseline_tau=-0.199569,
            ppo_b_baseline_state_noise_pos=0.0,
            ppo_b_baseline_state_noise_vel=0.0,
            ppo_b_case_source="scenario_library",
            ppo_b_library_path=MAIN_LIB,
            ppo_b_prob_good=0.0,
            ppo_b_prob_savable=1.0,
            ppo_b_prob_bad=0.0,
            ppo_b_eval_use_same_distribution=True,
            ppo_b_noise_theta_deg=0.0,
            ppo_b_noise_tli_dir_deg=0.0,
            ppo_b_noise_tli_dv_kms=0.0,
            ppo_b_use_fixed_index=True,
            ppo_b_fixed_index=MAIN_CASE_IDX,
            ppo_b_fixed_state_noise_pos=0.0,
            ppo_b_fixed_state_noise_vel=0.0,
            dv_noise_sigma_tli=0.0,
            dv_noise_sigma_mcc=0.0,
            use_manual_log_std=False,
            manual_log_std_value=0.0,
        ),
        CurriculumStage(
            name="ppo_b_dv_refine_final",
            reward_weights=RewardWeights(
                w_flyby=10.0,
                w_velocity=0.0,
                w_dv=15.0,
                w_return=100.0,
                w_budget=40.0,
                w_escape=60.0,
                w_earth_crash=150.0,
                w_moon_crash=20.0,
                w_postflyby_earth_crash=50.0,
                w_invalid_preflyby_earth_return=60.0,
            ),
            entropy_coef=0.001,
            timesteps=100_000,
            trainer_mode="ppo_b_library",
            tli_control_mode="full",
            mcc_enabled=True,
            tli_only_mode=False,
            reward_after_tli_ballistic_enabled=False,
            spawn_theta_limit_enabled=False,
            spawn_theta_min=0.0,
            spawn_theta_max=6.283185307179586,
            ppo_b_baseline_theta=4.7181,
            ppo_b_baseline_ax=0.9751105905,
            ppo_b_baseline_ay=-0.4365218282,
            ppo_b_baseline_tau=-0.199569,
            ppo_b_baseline_state_noise_pos=0.0,
            ppo_b_baseline_state_noise_vel=0.0,
            ppo_b_case_source="scenario_library",
            ppo_b_library_path=MAIN_LIB,
            ppo_b_prob_good=0.0,
            ppo_b_prob_savable=1.0,
            ppo_b_prob_bad=0.0,
            ppo_b_eval_use_same_distribution=True,
            ppo_b_noise_theta_deg=0.0,
            ppo_b_noise_tli_dir_deg=0.0,
            ppo_b_noise_tli_dv_kms=0.0,
            ppo_b_use_fixed_index=True,
            ppo_b_fixed_index=MAIN_CASE_IDX,
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
            "generate_mcc_eval_plot": True,
        },
        "env": {
            # example:
            "add_staged_tli_obs": False,
        },
        "reward": {
            # known-good PPO-B values
            "dv_budget": 0.1,
            "dv_scale": PPOB_DV_REF_ND,
            "beta_distance_flyby": 3.5,
            "r0_distance_flyby": 0.1,
            "beta_distance_return": 3.5,
            "r0_distance_return": 0.1,
        },
    }

    return curriculum, overrides