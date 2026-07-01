"""
strict_nominal_impulse_optimizer.py

Strict nominal TLI + MCC impulse optimizer for the Earth-Moon CR3BP project.

This is a full replacement for the grid-search nominal-action script.
It follows the philosophy of your old global_mcc_impulse_optimizer.py:

1) optimize only over one impulse magnitude and direction;
2) propagate the resulting trajectory directly;
3) only accept candidates that complete the event sequence:
   - enter lunar flyby radius,
   - pass lunar periapsis,
   - exit lunar flyby radius,
   - enter Earth-return corridor,
   - exit the return corridor outward,
   - no Earth impact,
   - no Moon impact,
   - no escape.

TLI is handled as one ballistic impulse from LEO.
MCC is handled as one impulse from the chosen handoff-library state.

Place this file in the same folder as:
    config.py
    cr3bp_env_v4.py
    rough_scenario_classification/

Run:
    python strict_nominal_impulse_optimizer.py

Outputs:
    sensitivity analysis/strict_nominal_optimizer_<timestamp>/
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

from config import RUN, CR3BPConfig, RewardConfig, RewardWeights
from cr3bp_env_v4 import (
    CR3BPFreeReturnEnv,
    SeanStyleReward,
    rk4_step,
    earth_moon_positions,
    radial_velocity_about_point,
    kms_to_nondim_dv,
    cr3bp_vstar_kms,
)


# ============================================================
# USER SETTINGS
# ============================================================

TAG = "strict_nominal_optimizer"

# --- TLI search ---
TLI_THETA_RAD = 4.04056

# X[0] = TLI delta-v magnitude [km/s]
# X[1] = burn angle offset from local prograde [deg]
TLI_BOUNDS = [
    (3.05, 3.30),
    (-35.0, 15.0),
]

# --- MCC search ---
MCC_LIBRARY_PATH = "rough_scenario_classification/ppob_handoff_states_30min.npz"
MCC_LIBRARY_INDEX = 65

# X[0] = MCC delta-v magnitude [m/s]
# X[1] = absolute burn direction in rotating frame [deg]
# For a faster local search around the known good region, use [(15, 35), (0, 40)].
MCC_BOUNDS = [
    (0.0, 60.0),
    (0.0, 360.0),
]

# Quick:     popsize=10, maxiter=40
# Medium:    popsize=20, maxiter=100
# Overnight: popsize=35, maxiter=250
POPSIZE_TLI = 12
MAXITER_TLI = 50

POPSIZE_MCC = 12
MAXITER_MCC = 50

POLISH = True
SEED = 999

# The old MCC optimizer used dt=0.0005. This is usually fast and stable enough.
FIXED_DT_ND = 0.0005
T_END_ND = 4.0
STORE_EVERY = 2
ESCAPE_RE_ND = 2.0
VERBOSE_BEST = True


RETURN_UPPER_ND = 0.05
RETURN_LOWER_MARGIN_ND = 0.001
RETURN_UPPER_MARGIN_ND = 0.001


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EventResult:
    valid: bool
    mode: str
    dv_mps: float
    angle_deg: float
    angle_offset_from_prograde_deg: Optional[float]
    score: float
    entered_flyby: bool
    passed_lunar_periapsis: bool
    exited_flyby: bool
    entered_corridor: bool
    exited_corridor_outward: bool
    earth_impact: bool
    moon_impact: bool
    escape: bool
    timeout: bool
    min_rM_nd: float
    min_rE_postflyby_nd: float
    corridor_dist_nd: float
    t_final_nd: float
    n_steps: int
    reason: str


class OptimizerState:
    def __init__(self, name: str):
        self.name = name
        self.best_score = np.inf
        self.best_result: Optional[EventResult] = None
        self.best_traj: Optional[np.ndarray] = None
        self.best_t: Optional[np.ndarray] = None
        self.best_markers: Dict[str, Any] = {}
        self.evals = 0


# ============================================================
# BASIC HELPERS
# ============================================================

def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite(x: Any, default: float = 1e9) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    if not np.isfinite(y):
        return default
    return y


def wrap_0_360_deg(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def make_base_cfg() -> CR3BPConfig:
    return CR3BPConfig()


def make_reward_model() -> SeanStyleReward:
    return SeanStyleReward(RewardConfig(), RewardWeights())


def earth_moon_cfg_values(cfg: CR3BPConfig) -> Dict[str, float]:
    return {
        "mu": float(getattr(cfg, "mu", 0.0121505856)),

        # Optimizer acceptance radii
        "r_moon_flyby": 0.050,
        "rp_min": float(getattr(cfg, "rp_min", 0.020)),
        "rp_max": 0.050,

        # Physical impact radii
        "r_earth_impact": float(getattr(cfg, "r_earth_impact", 0.017)),
        "r_moon_impact": float(getattr(cfg, "r_moon_impact", 0.0045)),
        "r0_earth": float(getattr(cfg, "r0_earth", 0.0176)),
    }


def get_dt() -> float:
    return float(FIXED_DT_ND)


def local_prograde_angle_for_tli(theta_rad: float, cfg: CR3BPConfig) -> Tuple[np.ndarray, float]:
    env = CR3BPFreeReturnEnv(cfg, seed=SEED, reward_model=make_reward_model())
    state0 = env._build_leo_state_from_theta(float(theta_rad))
    vx, vy = float(state0[2]), float(state0[3])
    prograde = math.atan2(vy, vx)
    return state0, prograde


def load_mcc_handoff_state(library_path: str, index: int) -> np.ndarray:
    path = Path(library_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"MCC handoff library not found:\n{path}")
    data = np.load(path, allow_pickle=False)
    if "state_handoff" not in data:
        raise KeyError(f"Expected 'state_handoff' in {path}")
    states = np.asarray(data["state_handoff"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"state_handoff must have shape (N,4), got {states.shape}")
    index = int(index)
    if index < 0 or index >= len(states):
        raise IndexError(f"MCC_LIBRARY_INDEX={index} out of range for {len(states)} cases")
    return states[index].copy()


def corridor_distance(rE: float, rp_min: float, rp_max: float) -> float:
    if rE < rp_min:
        return float(rp_min - rE)
    if rE > rp_max:
        return float(rE - rp_max)
    return 0.0


# ============================================================
# STRICT EVENT PROPAGATOR
# ============================================================

def propagate_and_classify(
    state0: np.ndarray,
    dv_vec_nd: np.ndarray,
    cfg: CR3BPConfig,
    mode: str,
    dv_mps: float,
    angle_deg: float,
    angle_offset_from_prograde_deg: Optional[float] = None,
    t0: float = 0.0,
) -> Tuple[EventResult, np.ndarray, np.ndarray, Dict[str, Any]]:
    vals = earth_moon_cfg_values(cfg)
    mu = vals["mu"]
    r_moon_flyby = vals["r_moon_flyby"]
    r_earth_impact = vals["r_earth_impact"]
    r_moon_impact = vals["r_moon_impact"]
    rp_min = vals["rp_min"]
    rp_max = vals["rp_max"]

    rE_pos, rM_pos = earth_moon_positions(mu)

    s = np.asarray(state0, dtype=np.float64).copy()
    s[2:4] += np.asarray(dv_vec_nd, dtype=np.float64).reshape(2,)

    dt = get_dt()
    t = float(t0)
    t_end = float(T_END_ND)

    traj: List[np.ndarray] = []
    thist: List[float] = []

    entered_flyby = False
    passed_lunar_periapsis = False
    exited_flyby = False
    entered_corridor = False
    exited_corridor_outward = False

    earth_impact = False
    moon_impact = False
    escape = False
    timeout = False

    min_rM = np.inf
    min_rE_postflyby = np.inf
    best_corridor_dist = np.inf

    prev_rM: Optional[float] = None
    rM_increasing_count = 0

    markers = {
        "flyby_entry_point": None,
        "flyby_exit_point": None,
        "corridor_entry_point": None,
        "corridor_exit_point": None,
    }

    n_steps = 0
    reason = "timeout"

    while t < t_end:
        if n_steps % max(1, int(STORE_EVERY)) == 0:
            traj.append(s.copy())
            thist.append(t)

        s = rk4_step(mu, s, dt)
        t += dt
        n_steps += 1

        pos = s[:2].astype(np.float64)
        vel = s[2:4].astype(np.float64)
        rE_vec = pos - rE_pos
        rM_vec = pos - rM_pos
        rE = float(np.linalg.norm(rE_vec))
        rM = float(np.linalg.norm(rM_vec))

        min_rM = min(min_rM, rM)

        # Impacts always fail before any success condition is accepted.
        if rE <= r_earth_impact:
            earth_impact = True
            reason = "earth_impact"
            break
        if rM <= r_moon_impact:
            moon_impact = True
            reason = "moon_impact"
            break
        if rE >= float(ESCAPE_RE_ND):
            escape = True
            reason = "escape"
            break

        # Lunar flyby: enter, pass periapsis, exit.
        if (not entered_flyby) and rM <= r_moon_flyby:
            entered_flyby = True
            markers["flyby_entry_point"] = pos.copy()

        if entered_flyby and prev_rM is not None:
            if rM > prev_rM:
                rM_increasing_count += 1
            else:
                rM_increasing_count = 0
            if rM_increasing_count >= 3:
                passed_lunar_periapsis = True

        if entered_flyby and passed_lunar_periapsis and (not exited_flyby) and rM > r_moon_flyby:
            exited_flyby = True
            markers["flyby_exit_point"] = pos.copy()

        prev_rM = rM

        # Earth return: only after a completed flyby.
        if exited_flyby:
            min_rE_postflyby = min(min_rE_postflyby, rE)
            best_corridor_dist = min(best_corridor_dist, corridor_distance(rE, rp_min, rp_max))

            return_lower = r_earth_impact + RETURN_LOWER_MARGIN_ND
            return_upper = RETURN_UPPER_ND - RETURN_UPPER_MARGIN_ND

            if (not entered_corridor) and (return_lower <= rE <= return_upper):
                entered_corridor = True
                markers["corridor_entry_point"] = pos.copy()

            if entered_corridor and (not exited_corridor_outward):
                vrE = radial_velocity_about_point(rE_vec, vel)
                if (rE > RETURN_UPPER_ND) and (vrE > 0.0):
                    exited_corridor_outward = True
                    markers["corridor_exit_point"] = pos.copy()
                    reason = "success"
                    break
                else:
                    timeout = True
                    reason = "timeout"

    valid = bool(
        entered_flyby
        and passed_lunar_periapsis
        and exited_flyby
        and entered_corridor
        and exited_corridor_outward
        and not earth_impact
        and not moon_impact
        and not escape
    )

    if not np.isfinite(min_rE_postflyby):
        min_rE_postflyby = np.inf
    if not np.isfinite(best_corridor_dist):
        best_corridor_dist = np.inf

    result = EventResult(
        valid=valid,
        mode=str(mode),
        dv_mps=float(dv_mps),
        angle_deg=float(angle_deg),
        angle_offset_from_prograde_deg=(None if angle_offset_from_prograde_deg is None else float(angle_offset_from_prograde_deg)),
        score=np.inf,
        entered_flyby=bool(entered_flyby),
        passed_lunar_periapsis=bool(passed_lunar_periapsis),
        exited_flyby=bool(exited_flyby),
        entered_corridor=bool(entered_corridor),
        exited_corridor_outward=bool(exited_corridor_outward),
        earth_impact=bool(earth_impact),
        moon_impact=bool(moon_impact),
        escape=bool(escape),
        timeout=bool(timeout),
        min_rM_nd=float(min_rM),
        min_rE_postflyby_nd=float(min_rE_postflyby),
        corridor_dist_nd=float(best_corridor_dist),
        t_final_nd=float(t),
        n_steps=int(n_steps),
        reason=str(reason),
    )

    return result, np.asarray(traj, dtype=np.float64), np.asarray(thist, dtype=np.float64), markers


# ============================================================
# OBJECTIVES
# ============================================================

def invalid_penalty(res: EventResult, cfg: CR3BPConfig) -> float:
    vals = earth_moon_cfg_values(cfg)
    r_moon_flyby = vals["r_moon_flyby"]
    rp_min = vals["rp_min"]

    score = 1e6
    if not res.entered_flyby:
        score += 300000.0
    if res.entered_flyby and not res.passed_lunar_periapsis:
        score += 200000.0
    if res.passed_lunar_periapsis and not res.exited_flyby:
        score += 150000.0
    if res.exited_flyby and not res.entered_corridor:
        score += 100000.0
    if res.entered_corridor and not res.exited_corridor_outward:
        score += 80000.0
    if res.earth_impact:
        score += 500000.0
    if res.moon_impact:
        score += 500000.0
    if res.escape:
        score += 250000.0

    if not res.entered_flyby:
        score += 10000.0 * max(0.0, res.min_rM_nd - r_moon_flyby)
    if res.exited_flyby and not res.entered_corridor:
        score += 50000.0 * finite(res.corridor_dist_nd, 10.0)
    if res.entered_corridor and not res.exited_corridor_outward:
        if np.isfinite(res.min_rE_postflyby_nd) and res.min_rE_postflyby_nd < rp_min:
            score += 100000.0 * (rp_min - res.min_rE_postflyby_nd)

    score += 0.01 * res.dv_mps
    return float(score)


def update_best(opt_state: OptimizerState, score: float, res: EventResult, traj: np.ndarray, thist: np.ndarray, markers: Dict[str, Any]) -> None:
    if score < opt_state.best_score:
        opt_state.best_score = float(score)
        opt_state.best_result = res
        opt_state.best_traj = traj
        opt_state.best_t = thist
        opt_state.best_markers = markers
        if VERBOSE_BEST:
            print(
                f"\n[{opt_state.name.upper()} BEST] eval={opt_state.evals} "
                f"valid={res.valid} dv={res.dv_mps:.3f} m/s "
                f"angle={res.angle_deg:.3f} deg score={score:.3f} "
                f"flyby={res.entered_flyby}/{res.passed_lunar_periapsis}/{res.exited_flyby} "
                f"corridor={res.entered_corridor}/{res.exited_corridor_outward} "
                f"earth={res.earth_impact} moon={res.moon_impact} escape={res.escape} "
                f"min_rM={res.min_rM_nd:.6g} min_rEpost={res.min_rE_postflyby_nd:.6g}"
            )


def make_tli_objective(cfg: CR3BPConfig, state0: np.ndarray, prograde_angle_rad: float, opt_state: OptimizerState):
    def objective(X: np.ndarray) -> float:
        opt_state.evals += 1
        dv_kms = float(X[0])
        offset_deg = float(X[1])
        angle_rad = float(prograde_angle_rad + math.radians(offset_deg))
        angle_deg = wrap_0_360_deg(math.degrees(angle_rad))
        dv_nd = kms_to_nondim_dv(dv_kms)
        dv_vec_nd = dv_nd * np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
        res, traj, thist, markers = propagate_and_classify(
            state0, dv_vec_nd, cfg, "tli", dv_kms * 1000.0, angle_deg, offset_deg, 0.0
        )
        score = float(res.dv_mps) if res.valid else invalid_penalty(res, cfg)
        res.score = float(score)
        update_best(opt_state, score, res, traj, thist, markers)
        return score
    return objective


def make_mcc_objective(cfg: CR3BPConfig, state0: np.ndarray, opt_state: OptimizerState):
    def objective(X: np.ndarray) -> float:
        opt_state.evals += 1
        dv_mps = float(X[0])
        angle_deg = wrap_0_360_deg(float(X[1]))
        angle_rad = math.radians(angle_deg)
        dv_nd = kms_to_nondim_dv(dv_mps / 1000.0)
        dv_vec_nd = dv_nd * np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
        res, traj, thist, markers = propagate_and_classify(
            state0, dv_vec_nd, cfg, "mcc", dv_mps, angle_deg, None, 0.0
        )
        score = float(res.dv_mps) if res.valid else invalid_penalty(res, cfg)
        res.score = float(score)
        update_best(opt_state, score, res, traj, thist, markers)
        return score
    return objective


# ============================================================
# SAVING AND PLOTTING
# ============================================================

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def plot_solution(out_dir: Path, name: str, cfg: CR3BPConfig, opt_state: OptimizerState) -> None:
    if opt_state.best_result is None or opt_state.best_traj is None:
        return
    res = opt_state.best_result
    traj = opt_state.best_traj
    vals = earth_moon_cfg_values(cfg)
    mu = vals["mu"]
    r_moon_flyby = vals["r_moon_flyby"]
    rp_min = vals["rp_min"]
    rp_max = vals["rp_max"]
    r_earth_impact = vals["r_earth_impact"]
    r_moon_impact = vals["r_moon_impact"]
    rE_pos, rM_pos = earth_moon_positions(mu)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    if traj.ndim == 2 and traj.shape[0] > 1:
        ax.plot(traj[:, 0], traj[:, 1], lw=1.3, label="Trajectory")
    ax.scatter([rE_pos[0]], [rE_pos[1]], s=150, label="Earth")
    ax.scatter([rM_pos[0]], [rM_pos[1]], s=80, label="Moon")
    ax.add_patch(plt.Circle((rE_pos[0], rE_pos[1]), r_earth_impact, fill=False, lw=1.0, linestyle=":"))
    ax.add_patch(plt.Circle((rM_pos[0], rM_pos[1]), r_moon_impact, fill=False, lw=1.0, linestyle=":"))
    ax.add_patch(plt.Circle((rM_pos[0], rM_pos[1]), r_moon_flyby, fill=False, lw=1.2, label="Flyby radius"))
    ax.add_patch(plt.Circle((rE_pos[0], rE_pos[1]), rp_min, fill=False, lw=1.2, linestyle="--", label="Return corridor inner"))
    ax.add_patch(plt.Circle((rE_pos[0], rE_pos[1]), rp_max, fill=False, lw=1.2, linestyle="--", label="Return corridor outer"))

    for key, label in [
        ("flyby_entry_point", "Flyby entry"),
        ("flyby_exit_point", "Flyby exit"),
        ("corridor_entry_point", "Corridor entry"),
        ("corridor_exit_point", "Corridor outward exit"),
    ]:
        p = opt_state.best_markers.get(key, None)
        if p is not None:
            p = np.asarray(p, dtype=float)
            if p.shape == (2,) and np.all(np.isfinite(p)):
                ax.scatter([p[0]], [p[1]], s=45, marker="x", label=label)

    valid_txt = "VALID" if res.valid else "INVALID"
    ax.set_title(f"{name.upper()} strict nominal reference: {valid_txt}\n"
                 f"dv={res.dv_mps:.3f} m/s, angle={res.angle_deg:.3f} deg, reason={res.reason}")
    ax.set_xlabel("x [nondim]")
    ax.set_ylabel("y [nondim]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")

    if traj.ndim == 2 and traj.shape[0] > 1:
        xy = traj[:, :2]
        good = np.all(np.isfinite(xy), axis=1)
        xy = xy[good]
        if len(xy) > 0:
            r = np.linalg.norm(xy, axis=1)
            xy2 = xy[r <= 2.5] if np.any(r <= 2.5) else xy
            xmin, ymin = np.nanmin(xy2, axis=0)
            xmax, ymax = np.nanmax(xy2, axis=0)
            pad = 0.08 * max(xmax - xmin, ymax - ymin, 0.1)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
    fig.tight_layout()
    fig.savefig(out_dir / f"best_{name}_trajectory.png")
    fig.savefig(out_dir / f"best_{name}_trajectory.pdf")
    plt.close(fig)


def save_solution(out_dir: Path, name: str, cfg: CR3BPConfig, opt_state: OptimizerState, extra: Dict[str, Any]) -> None:
    if opt_state.best_result is None:
        raise RuntimeError(f"No result for {name}")
    data = asdict(opt_state.best_result)
    data.update(extra)
    data["evals"] = int(opt_state.evals)
    save_json(out_dir / f"best_{name}_solution.json", data)
    if opt_state.best_traj is not None:
        np.savez(
            out_dir / f"best_{name}_trajectory_data.npz",
            trajectory=opt_state.best_traj,
            time=opt_state.best_t,
            markers=json.dumps(json_safe(opt_state.best_markers)),
            result=json.dumps(json_safe(data)),
        )
    plot_solution(out_dir, name, cfg, opt_state)


def write_summary(out_dir: Path, tli_state: OptimizerState, mcc_state: OptimizerState, elapsed_hr: float) -> None:
    lines = []
    lines.append("Strict nominal impulse optimizer summary")
    lines.append("=" * 60)
    lines.append(f"Elapsed time [h]: {elapsed_hr:.4f}")
    lines.append("")
    for name, st in [("TLI", tli_state), ("MCC", mcc_state)]:
        res = st.best_result
        lines.append(name)
        lines.append("-" * len(name))
        if res is None:
            lines.append("No result.")
        else:
            lines.append(f"Valid                      : {res.valid}")
            lines.append(f"DV [m/s]                   : {res.dv_mps:.6f}")
            lines.append(f"Angle [deg]                : {res.angle_deg:.6f}")
            if res.angle_offset_from_prograde_deg is not None:
                lines.append(f"Offset from prograde [deg] : {res.angle_offset_from_prograde_deg:.6f}")
            lines.append(f"Score                      : {res.score:.6f}")
            lines.append(f"Reason                     : {res.reason}")
            lines.append(f"Entered flyby              : {res.entered_flyby}")
            lines.append(f"Passed lunar periapsis     : {res.passed_lunar_periapsis}")
            lines.append(f"Exited flyby               : {res.exited_flyby}")
            lines.append(f"Entered corridor           : {res.entered_corridor}")
            lines.append(f"Exited corridor outward    : {res.exited_corridor_outward}")
            lines.append(f"Earth impact               : {res.earth_impact}")
            lines.append(f"Moon impact                : {res.moon_impact}")
            lines.append(f"Escape                     : {res.escape}")
            lines.append(f"min rM [nd]                : {res.min_rM_nd:.9g}")
            lines.append(f"min rE postflyby [nd]      : {res.min_rE_postflyby_nd:.9g}")
            lines.append(f"corridor distance [nd]     : {res.corridor_dist_nd:.9g}")
            lines.append(f"Evaluations                : {st.evals}")
        lines.append("")
    (out_dir / "optimization_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# OPTIMIZATION ENTRY POINTS
# ============================================================

def optimize_tli(out_dir: Path) -> OptimizerState:
    print("\n" + "=" * 70)
    print("STRICT TLI IMPULSE OPTIMIZER")
    print("=" * 70)
    cfg = make_base_cfg()
    state0, prograde = local_prograde_angle_for_tli(TLI_THETA_RAD, cfg)
    print(f"TLI theta            : {TLI_THETA_RAD:.8f} rad")
    print(f"Local prograde angle : {wrap_0_360_deg(math.degrees(prograde)):.3f} deg")
    print(f"TLI bounds           : {TLI_BOUNDS}")
    print(f"popsize/maxiter      : {POPSIZE_TLI}/{MAXITER_TLI}")
    st = OptimizerState("tli")
    objective = make_tli_objective(cfg, state0, prograde, st)
    result = differential_evolution(
        objective,
        bounds=TLI_BOUNDS,
        popsize=int(POPSIZE_TLI),
        maxiter=int(MAXITER_TLI),
        polish=bool(POLISH),
        seed=int(SEED),
        updating="immediate",
        workers=1,
        tol=1e-7,
    )
    print("\nTLI optimizer finished")
    print(f"scipy result x = {result.x}")
    print(f"scipy fun      = {result.fun}")
    save_solution(out_dir, "tli", cfg, st, {
        "theta_rad": TLI_THETA_RAD,
        "bounds": TLI_BOUNDS,
        "scipy_x": result.x.tolist(),
        "scipy_fun": float(result.fun),
    })
    return st


def optimize_mcc(out_dir: Path) -> OptimizerState:
    print("\n" + "=" * 70)
    print("STRICT MCC IMPULSE OPTIMIZER")
    print("=" * 70)
    cfg = make_base_cfg()
    state0 = load_mcc_handoff_state(MCC_LIBRARY_PATH, MCC_LIBRARY_INDEX)
    print(f"MCC library path     : {MCC_LIBRARY_PATH}")
    print(f"MCC library index    : {MCC_LIBRARY_INDEX}")
    print(f"MCC bounds           : {MCC_BOUNDS}")
    print(f"popsize/maxiter      : {POPSIZE_MCC}/{MAXITER_MCC}")
    st = OptimizerState("mcc")
    objective = make_mcc_objective(cfg, state0, st)
    result = differential_evolution(
        objective,
        bounds=MCC_BOUNDS,
        popsize=int(POPSIZE_MCC),
        maxiter=int(MAXITER_MCC),
        polish=bool(POLISH),
        seed=int(SEED + 1),
        updating="immediate",
        workers=1,
        tol=1e-7,
    )
    print("\nMCC optimizer finished")
    print(f"scipy result x = {result.x}")
    print(f"scipy fun      = {result.fun}")
    save_solution(out_dir, "mcc", cfg, st, {
        "library_path": MCC_LIBRARY_PATH,
        "library_index": MCC_LIBRARY_INDEX,
        "bounds": MCC_BOUNDS,
        "scipy_x": result.x.tolist(),
        "scipy_fun": float(result.fun),
    })
    return st


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_dir / "sensitivity analysis" / f"{TAG}_{timestamp_str()}")
    cfg = make_base_cfg()
    print("\nStrict nominal impulse optimizer")
    print(f"Output directory: {out_dir}")
    print(f"CR3BP V* = {cr3bp_vstar_kms():.9f} km/s")
    print(f"Propagation dt [nd] = {get_dt():.8g}")
    print(f"T_END_ND = {T_END_ND}")
    print("Mission radii:")
    for k, v in earth_moon_cfg_values(cfg).items():
        print(f"  {k:>16s} = {v}")

    start = time.time()
    tli_state = optimize_tli(out_dir)
    mcc_state = optimize_mcc(out_dir)
    elapsed_hr = (time.time() - start) / 3600.0
    write_summary(out_dir, tli_state, mcc_state, elapsed_hr)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Elapsed time [h]: {elapsed_hr:.4f}")
    print(f"Summary: {out_dir / 'optimization_summary.txt'}")
    for name, st in [("TLI", tli_state), ("MCC", mcc_state)]:
        res = st.best_result
        if res is not None:
            print(f"{name}: valid={res.valid}, dv={res.dv_mps:.3f} m/s, angle={res.angle_deg:.3f} deg, reason={res.reason}")


if __name__ == "__main__":
    main()
