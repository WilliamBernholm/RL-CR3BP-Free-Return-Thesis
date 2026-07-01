"""
============================================================
PATCHED-CONIC FREE-RETURN BASELINE TEST
============================================================

Purpose
-------
This script creates a first analytical / semi-analytical
free-return baseline for your CR3BP RL project.

It does NOT train a policy.

It:
1. Builds a 400 km circular LEO initial condition.
2. Computes a patched-conic TLI magnitude estimate.
3. Computes an initial phase-angle estimate.
4. Scans phase angle, TLI magnitude, and TLI direction.
5. Propagates each candidate in your existing CR3BP dynamics.
6. Selects the best free-return-like result.
7. Plots the best trajectory using your current plotting code.
8. Saves the best trajectory and scan table to disk.

Important
---------
Patched conics gives a first analytical baseline, not an exact
CR3BP free-return solution. The scan around the patched-conic
guess is what makes it useful inside your actual environment.

Place this file in the same folder as:
- cr3bp_env_v4.py
- cr3bp_plotting_v4.py
- config.py

Then run:
    python patched_conic_free_return_baseline.py
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import CR3BPConfig, RewardConfig, RewardWeights
from cr3bp_env_v4 import (
    CR3BPFreeReturnEnv,
    SeanStyleReward,
    cr3bp_vstar_kms,
    kms_to_nondim_dv,
    minutes_to_nondim_time,
    nondim_time_to_minutes,
    earth_moon_positions,
    dist_to_primaries,
    rk4_step,
)

from cr3bp_plotting_v4 import plot_trajectory


# ============================================================
# 1. Physical constants for the analytical patched-conic seed
# ============================================================

MU_EARTH_KM3_S2 = 398600.4418
MU_MOON_KM3_S2 = 4902.800066
R_EARTH_KM = 6378.1363
R_MOON_KM = 1737.4
EARTH_MOON_DISTANCE_KM = 384400.0

LEO_ALTITUDE_KM = 400.0
ENTRY_ALTITUDE_KM = 120.0

# A simple target lunar flyby altitude. We scan around the TLI,
# not around this value directly, but it is useful for reporting.
TARGET_PERILUNE_ALTITUDE_KM = 800.0


# ============================================================
# 2. User-tunable scan settings
# ============================================================

@dataclass
class ScanSettings:
    # The analytical seed will usually be around 3.08 km/s for 400 km LEO.
    # We scan a little wider because CR3BP free-return geometry is sensitive.
    dv_min_kms: float = 3.02
    dv_max_kms: float = 3.22
    dv_count: int = 41

    # Phase angle = Moon angle minus spacecraft angle at TLI.
    # For Apollo-like translunar transfers, useful values are often
    # around 120 to 140 degrees depending on transfer time.
    phase_min_deg: float = 105.0
    phase_max_deg: float = 145.0
    phase_count: int = 81

    # Direction offset from pure prograde tangential TLI.
    # alpha = 0 deg means pure tangential prograde.
    direction_min_deg: float = -5.0
    direction_max_deg: float = 5.0
    direction_count: int = 11

    # Propagation length.
    # Earth-Moon CR3BP nondimensional time is based on lunar mean motion.
    # 10 days is enough to see lunar encounter and early return for many cases.
    propagation_days: float = 10.0

    # RK4 step size for this standalone scan.
    # Smaller is more accurate but slower.
    rk4_step_minutes: float = 10.0

    # Only keep trajectory samples every N integration steps.
    # This keeps memory low during large scans.
    store_every: int = 4

    # Objective weights.
    # The scan tries to find:
    # - close lunar flyby
    # - Earth return near entry interface after flyby
    # - reasonable TLI magnitude near the analytical seed
    w_moon: float = 1.0
    w_return: float = 4.0
    w_dv: float = 0.2


# ============================================================
# 3. Patched-conic first guess
# ============================================================

def patched_conic_hohmann_seed(
    leo_altitude_km: float = LEO_ALTITUDE_KM,
    r_moon_km: float = EARTH_MOON_DISTANCE_KM,
) -> Dict[str, float]:
    """
    Compute a simple Earth-centered patched-conic / Hohmann-like seed.

    This assumes:
    - circular LEO
    - impulsive tangential TLI
    - transfer ellipse with apogee near lunar distance

    This is not yet a full free-return. It is the analytical first guess
    for TLI magnitude and phase angle.
    """
    r0 = R_EARTH_KM + float(leo_altitude_km)
    r1 = float(r_moon_km)

    v_circ = math.sqrt(MU_EARTH_KM3_S2 / r0)

    a_transfer = 0.5 * (r0 + r1)
    v_perigee_transfer = math.sqrt(
        MU_EARTH_KM3_S2 * (2.0 / r0 - 1.0 / a_transfer)
    )

    dv_tli = v_perigee_transfer - v_circ

    # Time from perigee to apogee for this simple ellipse.
    tof_s = math.pi * math.sqrt(a_transfer**3 / MU_EARTH_KM3_S2)
    tof_days = tof_s / 86400.0

    # Moon mean motion.
    moon_period_s = 27.321661 * 86400.0
    n_moon = 2.0 * math.pi / moon_period_s

    # First phase-angle estimate:
    # Moon angle ahead of spacecraft at TLI.
    phase_rad = math.pi - n_moon * tof_s
    phase_deg = math.degrees((phase_rad + 2.0 * math.pi) % (2.0 * math.pi))

    return {
        "r0_km": r0,
        "v_circ_kms": v_circ,
        "a_transfer_km": a_transfer,
        "v_perigee_transfer_kms": v_perigee_transfer,
        "dv_tli_kms": dv_tli,
        "tof_days": tof_days,
        "phase_deg": phase_deg,
    }


# ============================================================
# 4. Environment setup
# ============================================================

def build_env_for_baseline() -> CR3BPFreeReturnEnv:
    """
    Build your CR3BP environment for a standalone ballistic test.

    We use the environment mostly as a convenient container for:
    - config values
    - nondimensional units
    - plotting geometry
    - Earth/Moon CR3BP constants

    The actual propagation below directly uses rk4_step(...).
    """
    cfg = CR3BPConfig()

    # Force 400 km LEO in nondimensional Earth-Moon units.
    cfg.r0_earth = (R_EARTH_KM + LEO_ALTITUDE_KM) / EARTH_MOON_DISTANCE_KM

    # Entry corridor.
    # Your default cfg.rp_min is around 0.0143, which is slightly above Earth radius.
    # Here we set the lower corridor near 120 km entry altitude.
    r_entry = (R_EARTH_KM + ENTRY_ALTITUDE_KM) / EARTH_MOON_DISTANCE_KM

    # Keep a broad upper corridor for "return to Earth vicinity".
    # You can tighten this later.
    cfg.rp_min = r_entry
    cfg.rp_max = 0.060

    # Keep long enough propagation horizon.
    cfg.t_max = minutes_to_nondim_time(ScanSettings().propagation_days * 24.0 * 60.0)

    # Make sure this is a PPO-A-like geometry from LEO.
    cfg.trainer_mode = "ppo_a"
    cfg.tli_control_mode = "full"
    cfg.mcc_enabled = False
    cfg.tli_only_mode = False
    cfg.reward_after_tli_ballistic_enabled = False

    reward_model = SeanStyleReward(
        RewardConfig(),
        RewardWeights(),
    )

    env = CR3BPFreeReturnEnv(cfg, seed=1, reward_model=reward_model)
    return env


# ============================================================
# 5. State construction and impulse application
# ============================================================

def build_leo_state_with_phase(env: CR3BPFreeReturnEnv, phase_deg: float) -> np.ndarray:
    """
    Build the initial LEO state for a desired phase angle.

    In the CR3BP rotating frame, the Moon is fixed on the +x side
    relative to Earth. Therefore:

        phase angle = theta_moon - theta_spacecraft

    Since theta_moon = 0 in the rotating frame,

        theta_spacecraft = -phase_angle

    Your environment already has _build_leo_state_from_theta(theta),
    which constructs the correct circular LEO state in rotating-frame
    coordinates.
    """
    phase_rad = math.radians(float(phase_deg))
    theta_sc = -phase_rad

    state = env._build_leo_state_from_theta(theta_sc)
    return np.asarray(state, dtype=np.float64)


def local_radial_tangential_unit_vectors(
    env: CR3BPFreeReturnEnv,
    state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return local Earth-centered radial and prograde tangential unit vectors
    in the rotating-frame coordinate axes.

    An impulsive delta-v vector has the same components in the rotating
    and inertial coordinate axes at that instant, so we can add it directly
    to the rotating-frame velocity state.
    """
    mu = float(env.cfg.mu)
    rE_pos, _ = earth_moon_positions(mu)

    pos = np.asarray(state[:2], dtype=np.float64)
    r_vec = pos - rE_pos
    r_norm = np.linalg.norm(r_vec)

    if r_norm < 1e-12:
        raise ValueError("Spacecraft is too close to Earth center to define radial direction.")

    r_hat = r_vec / r_norm
    t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)

    return r_hat, t_hat


def apply_tli_impulse(
    env: CR3BPFreeReturnEnv,
    state_leo: np.ndarray,
    dv_tli_kms: float,
    direction_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply an impulsive TLI burn.

    direction_offset_deg:
        0 deg  -> pure prograde tangential burn
        >0 deg -> rotate slightly toward outward radial direction
        <0 deg -> rotate slightly toward inward radial direction

    Returns:
        state_after_tli, dv_vec_rot
    """
    state = np.asarray(state_leo, dtype=np.float64).copy()

    r_hat, t_hat = local_radial_tangential_unit_vectors(env, state)

    alpha = math.radians(float(direction_offset_deg))
    burn_dir = math.cos(alpha) * t_hat + math.sin(alpha) * r_hat
    burn_dir = burn_dir / max(np.linalg.norm(burn_dir), 1e-12)

    dv_nd = kms_to_nondim_dv(float(dv_tli_kms))
    dv_vec_rot = dv_nd * burn_dir

    state[2:4] += dv_vec_rot

    return state, dv_vec_rot


# ============================================================
# 6. Standalone CR3BP propagation and scoring
# ============================================================

def simulate_candidate(
    env: CR3BPFreeReturnEnv,
    phase_deg: float,
    dv_tli_kms: float,
    direction_offset_deg: float,
    settings: ScanSettings,
    store_traj: bool = False,
) -> Dict[str, object]:
    """
    Build a TLI candidate, propagate it in the CR3BP, and evaluate it.

    This does not call env.step().
    It hijacks the same CR3BP state convention and RK4 dynamics directly.

    The return condition is measured as:
        minimum Earth distance AFTER closest lunar approach.

    That is a simple and robust diagnostic for a free-return-like path.
    """
    state0 = build_leo_state_with_phase(env, phase_deg)
    state, dv_vec_rot = apply_tli_impulse(
        env=env,
        state_leo=state0,
        dv_tli_kms=dv_tli_kms,
        direction_offset_deg=direction_offset_deg,
    )

    mu = float(env.cfg.mu)
    rE_pos, rM_pos = earth_moon_positions(mu)

    dt = minutes_to_nondim_time(settings.rk4_step_minutes)
    t_end = minutes_to_nondim_time(settings.propagation_days * 24.0 * 60.0)

    n_steps = int(math.ceil(t_end / dt))

    min_rM = np.inf
    min_rM_t = np.nan
    min_rM_idx = -1

    traj_list: List[np.ndarray] = []
    t_list: List[float] = []

    rE_hist = []
    rM_hist = []
    t_hist_small = []

    t = 0.0

    if store_traj:
        traj_list.append(state.copy())
        t_list.append(t)

    for k in range(n_steps):
        state = rk4_step(mu, state, dt)
        t += dt

        pos = state[:2]
        rE = float(np.linalg.norm(pos - rE_pos))
        rM = float(np.linalg.norm(pos - rM_pos))

        rE_hist.append(rE)
        rM_hist.append(rM)
        t_hist_small.append(t)

        if rM < min_rM:
            min_rM = rM
            min_rM_t = t
            min_rM_idx = k

        if store_traj and (k % settings.store_every == 0):
            traj_list.append(state.copy())
            t_list.append(t)

        # Stop early if we crash deep into Earth.
        if rE <= float(env.cfg.r_earth_impact):
            break

        # Stop if obviously escaped far away.
        if rE >= float(env.cfg.r_escape):
            break

    rE_hist_arr = np.asarray(rE_hist, dtype=np.float64)
    rM_hist_arr = np.asarray(rM_hist, dtype=np.float64)
    t_hist_arr = np.asarray(t_hist_small, dtype=np.float64)

    if min_rM_idx >= 0 and min_rM_idx + 1 < len(rE_hist_arr):
        postflyby_rE = rE_hist_arr[min_rM_idx + 1 :]
        min_rE_postflyby = float(np.min(postflyby_rE)) if len(postflyby_rE) > 0 else np.inf
    else:
        min_rE_postflyby = np.inf

    # Corridor distance in nondimensional units.
    if min_rE_postflyby < env.cfg.rp_min:
        return_corridor_miss = float(env.cfg.rp_min - min_rE_postflyby)
    elif min_rE_postflyby > env.cfg.rp_max:
        return_corridor_miss = float(min_rE_postflyby - env.cfg.rp_max)
    else:
        return_corridor_miss = 0.0

    moon_miss = abs(float(min_rM) - float(env.cfg.r_moon_impact + TARGET_PERILUNE_ALTITUDE_KM / EARTH_MOON_DISTANCE_KM))

    seed = patched_conic_hohmann_seed()
    dv_penalty = abs(float(dv_tli_kms) - float(seed["dv_tli_kms"]))

    objective = (
        settings.w_moon * moon_miss
        + settings.w_return * return_corridor_miss
        + settings.w_dv * dv_penalty / cr3bp_vstar_kms()
    )

    corridor_hit = bool(return_corridor_miss <= 0.0)
    flyby_close = bool(min_rM <= float(env.cfg.r_moon_flyby))

    result = {
        "phase_deg": float(phase_deg),
        "dv_tli_kms": float(dv_tli_kms),
        "direction_offset_deg": float(direction_offset_deg),
        "dv_vec_rot": dv_vec_rot,
        "state0_leo": state0,
        "state_after_tli": None,
        "min_rM": float(min_rM),
        "min_rM_km": float(min_rM * EARTH_MOON_DISTANCE_KM),
        "min_rM_alt_km": float(min_rM * EARTH_MOON_DISTANCE_KM - R_MOON_KM),
        "min_rM_time_days": float(nondim_time_to_minutes(min_rM_t) / (60.0 * 24.0)) if np.isfinite(min_rM_t) else np.nan,
        "min_rE_postflyby": float(min_rE_postflyby),
        "min_rE_postflyby_km": float(min_rE_postflyby * EARTH_MOON_DISTANCE_KM),
        "min_rE_postflyby_alt_km": float(min_rE_postflyby * EARTH_MOON_DISTANCE_KM - R_EARTH_KM),
        "return_corridor_miss": float(return_corridor_miss),
        "return_corridor_miss_km": float(return_corridor_miss * EARTH_MOON_DISTANCE_KM),
        "corridor_hit": corridor_hit,
        "flyby_close": flyby_close,
        "objective": float(objective),
        "rE_hist": rE_hist_arr if store_traj else None,
        "rM_hist": rM_hist_arr if store_traj else None,
        "t_hist_small": t_hist_arr if store_traj else None,
    }

    if store_traj:
        traj = np.asarray(traj_list, dtype=np.float64)
        t_hist = np.asarray(t_list, dtype=np.float64)
        result["traj"] = traj
        result["t_hist"] = t_hist
        result["terminal_marker"] = traj[-1, :2].copy() if len(traj) > 0 else np.zeros((0,))
    else:
        result["traj"] = None
        result["t_hist"] = None
        result["terminal_marker"] = None

    return result


def run_scan(env: CR3BPFreeReturnEnv, settings: ScanSettings) -> Tuple[Dict[str, object], np.ndarray]:
    """
    Scan around the patched-conic seed and return the best candidate.
    """
    seed = patched_conic_hohmann_seed()

    print("\n" + "=" * 90)
    print("PATCHED-CONIC FIRST GUESS")
    print("=" * 90)
    print(f"LEO radius                         : {seed['r0_km']:.3f} km")
    print(f"Circular LEO speed                 : {seed['v_circ_kms']:.6f} km/s")
    print(f"Transfer perigee speed             : {seed['v_perigee_transfer_kms']:.6f} km/s")
    print(f"Patched-conic TLI estimate         : {seed['dv_tli_kms']:.6f} km/s")
    print(f"Hohmann-like time of flight        : {seed['tof_days']:.3f} days")
    print(f"Hohmann-like phase estimate        : {seed['phase_deg']:.3f} deg")
    print("")
    print("Note: Apollo-like faster transfers usually need a larger phase angle")
    print("than the slow Hohmann-like estimate, so the scan covers a wide range.")

    phases = np.linspace(settings.phase_min_deg, settings.phase_max_deg, settings.phase_count)
    dvs = np.linspace(settings.dv_min_kms, settings.dv_max_kms, settings.dv_count)
    dirs = np.linspace(settings.direction_min_deg, settings.direction_max_deg, settings.direction_count)

    rows = []
    best = None

    total = len(phases) * len(dvs) * len(dirs)
    count = 0

    print("\n" + "=" * 90)
    print("SCANNING CANDIDATES")
    print("=" * 90)
    print(f"Total candidates: {total}")

    for phase in phases:
        for dv in dvs:
            for alpha in dirs:
                count += 1

                res = simulate_candidate(
                    env=env,
                    phase_deg=float(phase),
                    dv_tli_kms=float(dv),
                    direction_offset_deg=float(alpha),
                    settings=settings,
                    store_traj=False,
                )

                rows.append([
                    res["phase_deg"],
                    res["dv_tli_kms"],
                    res["direction_offset_deg"],
                    res["min_rM_km"],
                    res["min_rM_alt_km"],
                    res["min_rE_postflyby_km"],
                    res["min_rE_postflyby_alt_km"],
                    res["return_corridor_miss_km"],
                    1.0 if res["flyby_close"] else 0.0,
                    1.0 if res["corridor_hit"] else 0.0,
                    res["objective"],
                ])

                if best is None or res["objective"] < best["objective"]:
                    best = res

        print(f"  finished phase {phase:.2f} deg")

    scan_table = np.asarray(rows, dtype=np.float64)

    # Re-simulate the best candidate with trajectory storage enabled.
    best_full = simulate_candidate(
        env=env,
        phase_deg=best["phase_deg"],
        dv_tli_kms=best["dv_tli_kms"],
        direction_offset_deg=best["direction_offset_deg"],
        settings=settings,
        store_traj=True,
    )

    return best_full, scan_table


# ============================================================
# 7. Plot and save
# ============================================================

def make_burn_event_for_plot(best: Dict[str, object]) -> List[Dict[str, object]]:
    """
    Create a burn event compatible with your plot_trajectory(...) function.
    """
    state0 = np.asarray(best["state0_leo"], dtype=np.float64)
    dv_vec = np.asarray(best["dv_vec_rot"], dtype=np.float64)

    dv_mag = float(np.linalg.norm(dv_vec))

    return [
        {
            "kind": "PATCHED_CONIC_TLI",
            "time": 0.0,
            "step_idx": 0,
            "ax_raw": np.nan,
            "ay_raw": np.nan,
            "tau_raw": np.nan,
            "tau_true": np.nan,
            "pos_rot": state0[:2].copy(),
            "dv_vec_rot": dv_vec.copy(),
            "dv_mag": dv_mag,
        }
    ]


def save_outputs(
    env: CR3BPFreeReturnEnv,
    best: Dict[str, object],
    scan_table: np.ndarray,
    settings: ScanSettings,
) -> Path:
    out_dir = Path("patched_conic_baseline_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = np.asarray(best["traj"], dtype=np.float64)
    t_hist = np.asarray(best["t_hist"], dtype=np.float64)
    burn_events = make_burn_event_for_plot(best)

    title = (
        "Patched-conic baseline in CR3BP "
        f"(phase={best['phase_deg']:.2f} deg, "
        f"dv={best['dv_tli_kms']:.4f} km/s, "
        f"dir={best['direction_offset_deg']:.2f} deg)"
    )

    plot_path = out_dir / "best_patched_conic_baseline_rotating.png"

    plot_trajectory(
        env.cfg,
        traj,
        burns=None,
        burn_events=burn_events,
        ballistic_ref_traj=None,
        ballistic_terminal_marker=None,
        terminal_marker=best["terminal_marker"],
        title=title,
        out_path=str(plot_path),
    )

    npz_path = out_dir / "patched_conic_scan_results.npz"

    np.savez_compressed(
        npz_path,
        scan_table=scan_table,
        scan_columns=np.asarray([
            "phase_deg",
            "dv_tli_kms",
            "direction_offset_deg",
            "min_rM_km",
            "min_rM_alt_km",
            "min_rE_postflyby_km",
            "min_rE_postflyby_alt_km",
            "return_corridor_miss_km",
            "flyby_close",
            "corridor_hit",
            "objective",
        ]),
        best_traj=traj,
        best_t_hist=t_hist,
        best_phase_deg=np.asarray([best["phase_deg"]]),
        best_dv_tli_kms=np.asarray([best["dv_tli_kms"]]),
        best_direction_offset_deg=np.asarray([best["direction_offset_deg"]]),
        best_min_rM_km=np.asarray([best["min_rM_km"]]),
        best_min_rM_alt_km=np.asarray([best["min_rM_alt_km"]]),
        best_min_rE_postflyby_km=np.asarray([best["min_rE_postflyby_km"]]),
        best_min_rE_postflyby_alt_km=np.asarray([best["min_rE_postflyby_alt_km"]]),
        best_return_corridor_miss_km=np.asarray([best["return_corridor_miss_km"]]),
    )

    summary_path = out_dir / "best_patched_conic_baseline_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PATCHED-CONIC FREE-RETURN BASELINE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"phase_deg                 = {best['phase_deg']:.9f}\n")
        f.write(f"dv_tli_kms                = {best['dv_tli_kms']:.9f}\n")
        f.write(f"direction_offset_deg      = {best['direction_offset_deg']:.9f}\n")
        f.write(f"min_rM_km                 = {best['min_rM_km']:.9f}\n")
        f.write(f"min_rM_alt_km             = {best['min_rM_alt_km']:.9f}\n")
        f.write(f"min_rM_time_days          = {best['min_rM_time_days']:.9f}\n")
        f.write(f"min_rE_postflyby_km       = {best['min_rE_postflyby_km']:.9f}\n")
        f.write(f"min_rE_postflyby_alt_km   = {best['min_rE_postflyby_alt_km']:.9f}\n")
        f.write(f"return_corridor_miss_km   = {best['return_corridor_miss_km']:.9f}\n")
        f.write(f"flyby_close               = {best['flyby_close']}\n")
        f.write(f"corridor_hit              = {best['corridor_hit']}\n")
        f.write(f"objective                 = {best['objective']:.12e}\n")
        f.write("\n")
        f.write(f"output_plot               = {plot_path}\n")
        f.write(f"output_npz                = {npz_path}\n")

    print("\n" + "=" * 90)
    print("BEST RESULT")
    print("=" * 90)
    print(f"phase angle at TLI              : {best['phase_deg']:.6f} deg")
    print(f"TLI magnitude                   : {best['dv_tli_kms']:.6f} km/s")
    print(f"TLI direction offset            : {best['direction_offset_deg']:.6f} deg")
    print(f"closest Moon distance           : {best['min_rM_km']:.3f} km")
    print(f"closest Moon altitude           : {best['min_rM_alt_km']:.3f} km")
    print(f"time of closest Moon approach   : {best['min_rM_time_days']:.3f} days")
    print(f"post-flyby closest Earth dist   : {best['min_rE_postflyby_km']:.3f} km")
    print(f"post-flyby closest Earth alt    : {best['min_rE_postflyby_alt_km']:.3f} km")
    print(f"return corridor miss            : {best['return_corridor_miss_km']:.3f} km")
    print(f"flyby_close                     : {best['flyby_close']}")
    print(f"corridor_hit                    : {best['corridor_hit']}")
    print("")
    print(f"Saved plot                      : {plot_path}")
    print(f"Saved scan data                 : {npz_path}")
    print(f"Saved summary                   : {summary_path}")

    return out_dir


# ============================================================
# 8. Main
# ============================================================

def main():
    settings = ScanSettings()

    env = build_env_for_baseline()

    print("\n" + "=" * 90)
    print("ENVIRONMENT / UNIT CHECK")
    print("=" * 90)
    print(f"CR3BP mu                         : {env.cfg.mu:.15f}")
    print(f"L*                               : {EARTH_MOON_DISTANCE_KM:.3f} km")
    print(f"V*                               : {cr3bp_vstar_kms():.6f} km/s")
    print(f"400 km LEO radius nondim         : {env.cfg.r0_earth:.9f}")
    print(f"entry radius nondim              : {env.cfg.rp_min:.9f}")
    print(f"return corridor max nondim       : {env.cfg.rp_max:.9f}")
    print(f"Moon flyby bound nondim          : {env.cfg.r_moon_flyby:.9f}")
    print(f"propagation days                 : {settings.propagation_days:.3f}")
    print(f"RK4 step minutes                 : {settings.rk4_step_minutes:.3f}")

    best, scan_table = run_scan(env, settings)
    save_outputs(env, best, scan_table, settings)


if __name__ == "__main__":
    main()