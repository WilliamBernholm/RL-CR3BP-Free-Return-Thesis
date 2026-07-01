"""
Reward-variation comparison plots.

Reads:
    Final plotting/data/tli_reward_variation/*.npz
    Final plotting/data/mcc_reward_variation/*.npz

Each file should contain:
    eval_step
    eval_reward_mean
    eval_dv_mean
    eval_dv_std
    eval_preservation_rate
    eval_rescue_rate

Success rate is computed as:
    success_rate = preservation_rate + rescue_rate
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from style.thesis_style import ieee_title


ENABLE_SUCCESS_PLOT_TLI = True
ENABLE_SUCCESS_PLOT_MCC = True

TLI_SUCCESS_REWARD_MARGIN = 0.1


FINAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FINAL_ROOT.parent

sys.path.insert(0, str(FINAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from style.thesis_style import (
    apply_thesis_style,
    get_figsize,
    clean_axis,
    save_thesis_figure,
)


# ============================================================
# TOGGLES
# ============================================================

PLOT_REWARD = True
PLOT_DV = True
PLOT_SUCCESS = True

SHOW_DV_STD_BAND = False

STEP_SCALE = 1e6
DV_UNIT_SCALE = 1000.0

XLABEL = r"Training steps [$10^6$]"


# ============================================================
# HELPERS
# ============================================================

def find_npz_files(data_dir: Path):
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        print(f"[WARN] No .npz files found in {data_dir}")
    return files


def load_file(path: Path):
    data = np.load(path, allow_pickle=True)
    return {k: np.asarray(data[k], dtype=float) for k in data.files}


def label_from_filename(path: Path):
    name = path.stem

    name = name.replace("final_training_curves_", "")
    name = name.replace("final_training_curves", "final")
    name = name.replace("_", " ")

    return name



def find_matching_config(npz_path: Path):
    stem = npz_path.stem
    suffix = stem.replace("final_training_curves_", "")

    candidates = [
        npz_path.with_name(f"run_config_{suffix}.txt"),
        npz_path.with_suffix(".txt"),
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def parse_tli_stages(config_path: Path):
    """
    Parse curriculum stages from run_config_*.txt.

    Returns list of:
        {
            "timesteps": float,
            "w_flyby": float,
            "w_return": float,
        }
    """

    text = config_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    stages = []
    current = None

    for raw in text:
        line = raw.strip()

        if line.startswith("Stage "):
            if current is not None:
                stages.append(current)
            current = {}

        elif current is not None:
            if line.startswith("timesteps ="):
                current["timesteps"] = float(line.split("=", 1)[1].strip())

            elif line.startswith("w_flyby ="):
                current["w_flyby"] = float(line.split("=", 1)[1].strip())

            elif line.startswith("w_return ="):
                current["w_return"] = float(line.split("=", 1)[1].strip())

            elif line.startswith("w_dv ="):
                current["w_dv"] = float(line.split("=", 1)[1].strip())

    if current is not None:
        stages.append(current)

    return stages


def parse_global_config_value(config_path: Path, key: str, default=None):
    for raw in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()

        if line.startswith(f"{key} ="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return default

    return default


def make_tli_threshold_per_eval(npz_path: Path, data: dict):
    """
    Build stage-dependent inferred TLI success thresholds.

    Success criterion:

        reward >=
        ballistic_scale * (w_flyby + w_return)
        - w_dv * dv_mean
        - margin

    """

    if "eval_step" not in data or "eval_reward_mean" not in data:
        return None

    config_path = find_matching_config(npz_path)

    if config_path is None:
        print(f"[WARN] No matching run_config found for {npz_path.name}")
        return None

    scale = parse_global_config_value(
        config_path,
        "tli_ballistic_scale",
        default=0.7,
    )

    stages = parse_tli_stages(config_path)

    if not stages:
        print(f"[WARN] No curriculum stages parsed from {config_path.name}")
        return None

    eval_steps = np.asarray(data["eval_step"], dtype=float)

    thresholds = np.zeros_like(eval_steps, dtype=float)

    dv_mean = np.asarray(
        data.get("eval_dv_mean", np.zeros_like(eval_steps)),
        dtype=float,
    )

    cumulative_end = 0.0

    for stage in stages:

        stage_steps = stage.get("timesteps", None)

        if stage_steps is None:
            continue

        cumulative_end += stage_steps

        w_flyby = stage.get("w_flyby", 0.0)
        w_return = stage.get("w_return", 0.0)
        w_dv = stage.get("w_dv", 0.0)

        stage_threshold = (
            scale * (w_flyby + w_return)
            - w_dv * dv_mean
            - TLI_SUCCESS_REWARD_MARGIN
        )

        mask = eval_steps <= cumulative_end
        unset = thresholds == 0.0

        assign_mask = mask & unset

        thresholds[assign_mask] = stage_threshold[assign_mask]

    # fallback for any remaining unset values
    unset = thresholds == 0.0

    if np.any(unset):

        last = stages[-1]

        last_w_flyby = last.get("w_flyby", 0.0)
        last_w_return = last.get("w_return", 0.0)
        last_w_dv = last.get("w_dv", 0.0)

        thresholds[unset] = (
            scale * (last_w_flyby + last_w_return)
            - last_w_dv * dv_mean[unset]
            - TLI_SUCCESS_REWARD_MARGIN
        )

    return thresholds


def infer_tli_success_rate_from_reward(npz_path: Path, data: dict):
    thresholds = make_tli_threshold_per_eval(npz_path, data)

    if thresholds is None:
        return None

    rewards = np.asarray(data["eval_reward_mean"], dtype=float)
    success = rewards >= thresholds

    return success.astype(float)


def compute_success_rate(data: dict, npz_path: Path = None, mission_name: str = "mcc"):
    """
    TLI:
        inferred from reward magnitude and stage-dependent config threshold.

    MCC:
        directly from preservation + rescue.
    """

    if mission_name.lower() == "tli":
        if npz_path is None:
            return None
        return infer_tli_success_rate_from_reward(npz_path, data)

    preservation = data.get("eval_preservation_rate", None)
    rescue = data.get("eval_rescue_rate", None)

    if preservation is None or rescue is None:
        return None

    return np.clip(preservation + rescue, 0.0, 1.0)


def make_legend_label(path: Path, data: dict, mission_name: str):
    base = label_from_filename(path)

    success = compute_success_rate(
        data,
        npz_path=path,
        mission_name=mission_name,
    )

    if success is None:
        return base

    mean_success = np.nanmean(success)
    return f"{base} | mean SR={100.0 * mean_success:.1f}%"







def plot_quantity(data_dir: Path, output_dir: Path, mission_name: str, quantity: str):
    apply_thesis_style()

    files = find_npz_files(data_dir)
    if not files:
        return

    fig, ax = plt.subplots(figsize=get_figsize("double"))

    for path in files:
        data = load_file(path)

        if "eval_step" not in data:
            print(f"[SKIP] {path.name}: missing eval_step")
            continue

        x = data["eval_step"] / STEP_SCALE
        label = make_legend_label(path, data, mission_name)

        if quantity == "reward":
            if "eval_reward_mean" not in data:
                print(f"[SKIP] {path.name}: missing eval_reward_mean")
                continue
            y = data["eval_reward_mean"]
            ylabel = "Mean evaluation reward"
            title = f"PPO-{mission_name.upper()} reward-variation comparison"
            out_name = f"{mission_name}_reward_variation_reward"

        elif quantity == "dv":
            if "eval_dv_mean" not in data:
                print(f"[SKIP] {path.name}: missing eval_dv_mean")
                continue
            y = data["eval_dv_mean"] * DV_UNIT_SCALE
            ylabel = r"Mean $\Delta v$ usage [m/s]"
            title = f"PPO-{mission_name.upper()} reward-variation: mean Δv"
            out_name = f"{mission_name}_reward_variation_dv"

        elif quantity == "success":
            y = compute_success_rate(
                data,
                npz_path=path,
                mission_name=mission_name,
            )
            if y is None:
                print(f"[SKIP] {path.name}: missing success-rate keys")
                continue
            y = 100.0 * y
            ylabel = "Success rate [%]"
            title = f"PPO-{mission_name.upper()} reward-variation: success rate"
            out_name = f"{mission_name}_reward_variation_success"

        else:
            raise ValueError(quantity)

        ax.plot(x, y, label=label)

        if quantity == "dv" and SHOW_DV_STD_BAND and "eval_dv_std" in data:
            std = data["eval_dv_std"] * DV_UNIT_SCALE
            ax.fill_between(x, y - std, y + std, alpha=0.12, linewidth=0)

    clean_axis(ax, grid=True)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(ieee_title(title))

    if quantity == "success":
        ax.set_ylim(-2, 102)

    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    save_thesis_figure(fig, output_dir / out_name)
    plt.close(fig)

    print(f"[OK] Saved {out_name}")


# ============================================================
# ENTRY POINTS
# ============================================================

def main_tli(data_dir: Path, output_dir: Path):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if PLOT_REWARD:
        plot_quantity(data_dir, output_dir, "tli", "reward")

    if PLOT_DV:
        plot_quantity(data_dir, output_dir, "tli", "dv")

    if PLOT_SUCCESS and ENABLE_SUCCESS_PLOT_TLI:
        plot_quantity(data_dir, output_dir, "tli", "success")


def main_mcc(data_dir: Path, output_dir: Path):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if PLOT_REWARD:
        plot_quantity(data_dir, output_dir, "mcc", "reward")

    if PLOT_DV:
        plot_quantity(data_dir, output_dir, "mcc", "dv")

    if PLOT_SUCCESS and ENABLE_SUCCESS_PLOT_MCC:
        plot_quantity(data_dir, output_dir, "mcc", "success")


def main(data_dir=None, output_dir=None):
    if output_dir is None:
        output_dir = FINAL_ROOT / "outputs" / "thesis_ready"

    main_tli(FINAL_ROOT / "data" / "tli_reward_variation", output_dir)
    main_mcc(FINAL_ROOT / "data" / "mcc_reward_variation", output_dir)


if __name__ == "__main__":
    main()