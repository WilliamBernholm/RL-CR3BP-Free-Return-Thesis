"""
Plot thesis-ready PPO training curves.

Reads every .npz file inside:
    Final plotting/data/ppo_tli_training/
    Final plotting/data/ppo_mcc_training/

Expected keys:
    eval_step
    eval_reward_mean
    eval_dv_mean
    eval_dv_std

Run through:
    python "Final plotting/scripts/plot_all.py"
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from style.thesis_style import (
    apply_thesis_style,
    get_figsize,
    clean_axis,
    save_thesis_figure,
    ieee_title,
)




# ============================================================
# SETTINGS
# ============================================================

SHOW_STD_BAND = True
DV_UNIT_SCALE = 1000.0          # km/s -> m/s if data is stored in km/s
STEP_SCALE = 1e6                # show x-axis in million training steps

REWARD_YLABEL = "Mean evaluation reward"
DV_YLABEL = r"Mean $\Delta v$ usage [m/s]"
XLABEL = r"Training steps [$10^6$]"

REWARD_TITLE_TLI = "PPO-TLI: Mean Evaluation Reward"
DV_TITLE_TLI = r"PPO-TLI: Mean $\Delta v$ Usage"

REWARD_TITLE_MCC = "PPO-MCC: Mean Evaluation Reward"
DV_TITLE_MCC = r"PPO-MCC: Mean $\Delta v$ Usage"


# ============================================================
# HELPERS
# ============================================================

def find_npz_files(data_dir: Path):
    files = sorted(Path(data_dir).glob("*.npz"))

    if not files:
        print(f"[WARN] No .npz files found in: {data_dir}")

    return files


def load_curve_file(path: Path):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    required = {
        "eval_step",
        "eval_reward_mean",
        "eval_dv_mean",
    }

    missing = required - keys
    if missing:
        raise KeyError(
            f"{path.name} is missing keys: {sorted(missing)}\n"
            f"Available keys: {sorted(keys)}"
        )

    out = {
        "step": np.asarray(data["eval_step"], dtype=float),
        "reward_mean": np.asarray(data["eval_reward_mean"], dtype=float),
        "dv_mean": np.asarray(data["eval_dv_mean"], dtype=float),
        "dv_std": None,
    }

    if "eval_dv_std" in keys:
        out["dv_std"] = np.asarray(data["eval_dv_std"], dtype=float)

    return out


def make_label(path: Path):
    stem = path.stem

    if stem == "final_training_curves":
        return "Final run"

    return stem.replace("_", " ")


def plot_reward_curve(data_dir: Path, output_dir: Path, mission_name: str):
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=get_figsize("single"))

    files = find_npz_files(data_dir)

    for path in files:
        curve = load_curve_file(path)

        x = curve["step"] / STEP_SCALE
        y = curve["reward_mean"]

        ax.plot(x, y, label=make_label(path))

    clean_axis(ax, grid=True)

    if mission_name == "tli":
        ax.set_title(ieee_title(ieee_title(REWARD_TITLE_TLI)))
        out_name = "ppo_tli_reward_curve"
    else:
        ax.set_title(ieee_title(ieee_title(REWARD_TITLE_MCC)))
        out_name = "ppo_mcc_reward_curve"

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(REWARD_YLABEL)

    if len(files) > 1:
        ax.legend(loc="best")

    save_thesis_figure(fig, output_dir / out_name)
    plt.close(fig)

    print(f"[OK] Saved reward curve: {out_name}")


def plot_dv_curve(data_dir: Path, output_dir: Path, mission_name: str):
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=get_figsize("single"))

    files = find_npz_files(data_dir)

    for path in files:
        curve = load_curve_file(path)

        x = curve["step"] / STEP_SCALE
        y = curve["dv_mean"] * DV_UNIT_SCALE

        label = make_label(path)
        ax.plot(x, y, label=label)

        if SHOW_STD_BAND and curve["dv_std"] is not None:
            std = curve["dv_std"] * DV_UNIT_SCALE
            ax.fill_between(x, y - std, y + std, alpha=0.18, linewidth=0)

    clean_axis(ax, grid=True)

    if mission_name == "tli":
        ax.set_title(ieee_title(DV_TITLE_TLI))
        out_name = "ppo_tli_dv_curve"
    else:
        ax.set_title(ieee_title(DV_TITLE_MCC))
        out_name = "ppo_mcc_dv_curve"

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(DV_YLABEL)

    if len(files) > 1:
        ax.legend(loc="best")

    save_thesis_figure(fig, output_dir / out_name)
    plt.close(fig)

    print(f"[OK] Saved dv curve: {out_name}")


# ============================================================
# ENTRY POINTS USED BY plot_all.py
# ============================================================

def main_tli(data_dir: Path, output_dir: Path, plot_reward=True, plot_dv=True):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if plot_reward:
        plot_reward_curve(data_dir, output_dir, mission_name="tli")

    if plot_dv:
        plot_dv_curve(data_dir, output_dir, mission_name="tli")


def main_mcc(data_dir: Path, output_dir: Path, plot_reward=True, plot_dv=True):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if plot_reward:
        plot_reward_curve(data_dir, output_dir, mission_name="mcc")

    if plot_dv:
        plot_dv_curve(data_dir, output_dir, mission_name="mcc")


# ============================================================
# DIRECT RUN DEBUG MODE
# ============================================================

if __name__ == "__main__":
    here = Path(__file__).resolve()
    final_root = here.parents[1]

    main_tli(
        final_root / "data" / "ppo_tli_training",
        final_root / "outputs" / "thesis_ready",
        plot_reward=True,
        plot_dv=True,
    )