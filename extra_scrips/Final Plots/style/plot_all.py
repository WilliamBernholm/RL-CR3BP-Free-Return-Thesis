"""
Master script for generating final thesis plots.

Run from project root:
    python "Final plotting/scripts/plot_all.py"
"""

from pathlib import Path
import sys



# ============================================================
# PATH SETUP
# ============================================================

FINAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FINAL_ROOT.parent

DATA_DIR = FINAL_ROOT / "data"
OUTPUT_DIR = FINAL_ROOT / "outputs"
THESIS_DIR = OUTPUT_DIR / "thesis_ready"
PROJECT_MODULES_DIR = FINAL_ROOT / "project_modules"

sys.path.insert(0, str(FINAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_MODULES_DIR))


# ============================================================
# TOGGLES
# ============================================================

# Training curves
RUN_TLI_TRAINING_CURVES = True
RUN_MCC_TRAINING_CURVES = True

# PPO diagnostics
RUN_TLI_PPO_METRICS = True
RUN_MCC_PPO_METRICS = True

# Trajectories
RUN_TLI_TRAJECTORY = True
RUN_MCC_TRAJECTORY = True

# Reward landscapes
RUN_REWARD_LANDSCAPES = True

# Sensitivity / success heatmaps
RUN_SENSITIVITY_SUCCESS = True

# Reward formulation comparison
RUN_REWARD_VARIATIONS = True


# Free return grid sweep, sucsess and lunar distance
RUN_GRID_SWEEP_FREE_RETURN = True




# ============================================================
# FOLDER SETUP
# ============================================================

def ensure_folder_structure():
    folders = [
        # Project module copies
        PROJECT_MODULES_DIR,

        # Training curve data
        DATA_DIR / "ppo_tli_training",
        DATA_DIR / "ppo_mcc_training",

        # PPO metrics use same files as training curves, but separate folders
        # are still allowed if you prefer.
        DATA_DIR / "ppo_metrics_tli",
        DATA_DIR / "ppo_metrics_mcc",

        # Trajectory archives
        DATA_DIR / "tli_trajectory",
        DATA_DIR / "mcc_trajectory",

        # Reward landscapes generated from project modules
        DATA_DIR / "reward_landscapes",

        # Sensitivity success heatmaps
        DATA_DIR / "sensitivity_tli",
        DATA_DIR / "sensitivity_mcc",

        # Reward formulation variations
        DATA_DIR / "tli_reward_variation",
        DATA_DIR / "mcc_reward_variation",

        # Grid sweep
        DATA_DIR / "grid_sweep_free_return",


        # Outputs
        OUTPUT_DIR,
        THESIS_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_folder_structure()

    print("\n" + "=" * 72)
    print("FINAL THESIS PLOTTING PIPELINE")
    print("=" * 72)
    print(f"Final plotting folder : {FINAL_ROOT}")
    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Project modules       : {PROJECT_MODULES_DIR}")
    print(f"Data folder           : {DATA_DIR}")
    print(f"Output folder         : {THESIS_DIR}")
    print("=" * 72 + "\n")

    # --------------------------------------------------------
    # Training curves: reward and delta-v
    # --------------------------------------------------------
    if RUN_TLI_TRAINING_CURVES:
        from scripts.plot_training_curves import main_tli as run
        run(
            DATA_DIR / "ppo_tli_training",
            THESIS_DIR,
            plot_reward=True,
            plot_dv=True,
        )

    if RUN_MCC_TRAINING_CURVES:
        from scripts.plot_training_curves import main_mcc as run
        run(
            DATA_DIR / "ppo_mcc_training",
            THESIS_DIR,
            plot_reward=True,
            plot_dv=True,
        )

    # --------------------------------------------------------
    # PPO metrics
    # --------------------------------------------------------
    if RUN_TLI_PPO_METRICS:
        from scripts.plot_ppo_metrics import main_tli as run
        run(DATA_DIR / "ppo_tli_training", THESIS_DIR)

    if RUN_MCC_PPO_METRICS:
        from scripts.plot_ppo_metrics import main_mcc as run
        run(DATA_DIR / "ppo_mcc_training", THESIS_DIR)

    # --------------------------------------------------------
    # Trajectories
    # --------------------------------------------------------
    if RUN_TLI_TRAJECTORY:
        from scripts.plot_trajectory_tli_mcc import main_tli as run
        run(DATA_DIR / "tli_trajectory", THESIS_DIR)

    if RUN_MCC_TRAJECTORY:
        from scripts.plot_trajectory_tli_mcc import main_mcc as run
        run(DATA_DIR / "mcc_trajectory", THESIS_DIR)

    # --------------------------------------------------------
    # Reward landscapes
    # --------------------------------------------------------
    if RUN_REWARD_LANDSCAPES:
        from scripts.plot_reward_landscapes import main as run
        run(DATA_DIR / "reward_landscapes", THESIS_DIR)

    # --------------------------------------------------------
    # Sensitivity success heatmaps
    # --------------------------------------------------------
    if RUN_SENSITIVITY_SUCCESS:
        from scripts.plot_sensitivity_analysis import main as run
        run(DATA_DIR, THESIS_DIR)

    # --------------------------------------------------------
    # Reward variation comparisons
    # --------------------------------------------------------
    if RUN_REWARD_VARIATIONS:
        from scripts.plot_reward_variations import main as run
        run(DATA_DIR, THESIS_DIR)


    # --------------------------------------------------------
    # Free-return grid sweep
    # --------------------------------------------------------
    if RUN_GRID_SWEEP_FREE_RETURN:
        from scripts.plot_grid_sweep_free_return import main as run
        run(
            DATA_DIR / "grid_sweep_free_return",
            THESIS_DIR,
        )


    print("\nDone.\n")


if __name__ == "__main__":
    main()