"""
Plot thesis-ready PPO diagnostic metrics.

Uses:
    ppo_step
    approx_kl
    clip_fraction
    policy_gradient_loss
    value_loss
    loss
    entropy_loss
    explained_variance
    std
    learning_rate

Outputs one stacked diagnostic figure per mission.
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


STEP_SCALE = 1e6
XLABEL = r"Training steps [$10^6$]"

CLIP_RANGE_DEFAULT = 0.15
SHOW_CLIP_RANGE = True


def find_npz_files(data_dir: Path):
    return sorted(Path(data_dir).glob("*.npz"))


def load_metrics(path: Path):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    required = [
        "ppo_step",
        "approx_kl",
        "clip_fraction",
        "policy_gradient_loss",
        "value_loss",
        "loss",
        "entropy_loss",
        "explained_variance",
        "std",
    ]

    missing = [k for k in required if k not in keys]
    if missing:
        raise KeyError(
            f"{path.name} is missing keys: {missing}\n"
            f"Available keys: {sorted(keys)}"
        )

    return {k: np.asarray(data[k], dtype=float) for k in required}


def plot_single_metrics_file(path: Path, output_dir: Path, mission_name: str):
    apply_thesis_style()

    m = load_metrics(path)
    x = m["ppo_step"] / STEP_SCALE

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(5.2, 7.0),
        sharex=True,
    )

    # --------------------------------------------------------
    # 1) Policy / clipping
    # --------------------------------------------------------
    ax = axes[0]
    ax.plot(x, m["approx_kl"], label=r"Approx. KL")
    ax.plot(x, m["clip_fraction"], label="Clip fraction")
    ax.plot(x, m["policy_gradient_loss"], label="Policy gradient loss")

    if SHOW_CLIP_RANGE:
        ax.axhline(
            CLIP_RANGE_DEFAULT,
            linestyle="--",
            linewidth=1.0,
            label="Clip range",
        )

    ax.set_ylabel("Policy")
    ax.set_title(ieee_title(f"PPO-{mission_name.upper()}: Training Diagnostics"))
    ax.legend(loc="best", ncol=2)
    clean_axis(ax, grid=True)

    # --------------------------------------------------------
    # 2) Value / fit
    # --------------------------------------------------------
    ax = axes[1]
    ax.plot(x, m["loss"], label="Total loss")
    ax.plot(x, m["value_loss"], label="Value loss")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", ncol=2)
    clean_axis(ax, grid=True)

    # --------------------------------------------------------
    # 3) Explained variance
    # --------------------------------------------------------
    ax = axes[2]
    ax.plot(x, m["explained_variance"], label="Explained variance")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Explained Variance")
    ax.legend(loc="best")
    clean_axis(ax, grid=True)

    # --------------------------------------------------------
    # 4) Entropy
    # --------------------------------------------------------
    ax = axes[3]
    ax.plot(x, m["entropy_loss"], label="Entropy loss")
    ax.set_ylabel("Entropy")
    ax.legend(loc="best")
    clean_axis(ax, grid=True)

    # --------------------------------------------------------
    # 5) Policy std
    # --------------------------------------------------------
    ax = axes[4]
    ax.plot(x, m["std"], label="Policy std")
    ax.set_ylabel("Std")
    ax.set_xlabel(XLABEL)
    ax.legend(loc="best")
    clean_axis(ax, grid=True)

    fig.tight_layout(h_pad=0.7)

    out_name = f"ppo_{mission_name.lower()}_metrics"
    save_thesis_figure(fig, output_dir / out_name)
    plt.close(fig)

    print(f"[OK] Saved PPO metrics: {out_name}")


def main_tli(data_dir: Path, output_dir: Path):
    files = find_npz_files(data_dir)

    if not files:
        print(f"[WARN] No PPO-TLI metric files found in {data_dir}")
        return

    plot_single_metrics_file(files[0], output_dir, mission_name="tli")


def main_mcc(data_dir: Path, output_dir: Path):
    files = find_npz_files(data_dir)

    if not files:
        print(f"[WARN] No PPO-MCC metric files found in {data_dir}")
        return

    plot_single_metrics_file(files[0], output_dir, mission_name="mcc")


if __name__ == "__main__":
    here = Path(__file__).resolve()
    final_root = here.parents[1]

    main_tli(
        final_root / "data" / "ppo_metrics_tli",
        final_root / "outputs" / "thesis_ready",
    )