"""
============================================================
CR3BP FREE-RETURN RL ENVIRONMENT
============================================================

This module implements the full planar Earth Moon CR3BP
reinforcement learning environment used for trajectory
optimization with PPO-based agents.

It provides:
- CR3BP dynamics and geometry (Earth Moon rotating frame)
- adaptive RK4 propagation with region-based refinement
- action decoding for impulsive burns (TLI and MCC)
- reset / spawn logic for multiple training regimes
- a configurable reward function for trajectory evaluation
- curriculum stage integration via config mapping
- unit conversion helpers (km/s <-> nondimensional)
- scenario-library support for PPO-B handoff states

------------------------------------------------------------
SUPPORTED TRAINING MODES
------------------------------------------------------------

PPO-A (TLI optimization):
- The spacecraft is initialized in a circular LEO around Earth
- The initial position is defined by a spawn angle (theta)
- The agent learns the translunar injection (TLI) burn:
  direction, magnitude, and timing (via tau)
- The trajectory is then propagated toward the Moon

PPO-B (MCC / correction optimization):
- The spacecraft is initialized from a known post-TLI state
- This state is loaded from a precomputed scenario library
  (handoff states generated from PPO-A or other sources)
- The agent applies a mid-course correction (MCC) burn
- The goal is to recover or refine trajectories toward a
  successful lunar flyby and Earth return

------------------------------------------------------------
MISSION STRUCTURE
------------------------------------------------------------

- Pre-TLI phase:
  fine phasing in LEO with small allowable burns

- TLI event:
  a committed burn that transitions the spacecraft from
  Earth orbit onto a translunar trajectory

- Post-TLI phase:
  long ballistic propagation with optional MCC intervention

- Terminal evaluation:
  trajectory is evaluated based on:
    - lunar flyby geometry
    - Earth return corridor
    - velocity constraints
    - delta-v usage
    - failure conditions (impact, escape, invalid return)

------------------------------------------------------------
REWARD MODEL
------------------------------------------------------------

The environment uses a configurable reward function that maps
trajectory outcomes and intermediate behavior to a scalar reward.

It includes contributions from:
- flyby quality (distance and geometry)
- return corridor accuracy
- velocity constraints at key events
- total delta-v usage and budget penalties
- terminal penalties (Earth/Moon impact, escape, invalid cases)

All weights and shaping parameters are controlled externally
via RewardConfig and RewardWeights.

------------------------------------------------------------
DESIGN INTENT
------------------------------------------------------------

This environment is designed to:
- support both TLI learning (PPO-A) and MCC correction (PPO-B)
- allow curriculum-based training with controlled difficulty
- maintain consistent action-to-physics mapping across stages
- provide detailed diagnostics for evaluation and debugging

============================================================
"""



from __future__ import annotations

import math
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    RUN,
    RunConfig,
    RewardConfig,
    RewardWeights,
    CurriculumStage,
    CR3BPConfig,
    ppo_rollout_block_size,
)

import numpy as np
import gymnasium as gym
from gymnasium import spaces



def cr3bp_vstar_kms() -> float:
    """
    Characteristic velocity for the Earth-Moon CR3BP nondimensionalization:
        V* = L* / T*
    returned in km/s.
    """
    return float(RUN.cr3bp_Lstar_km) / float(RUN.cr3bp_Tstar_s)


def kms_to_nondim_dv(dv_kms: float) -> float:
    """
    Convert physical delta-v in km/s to CR3BP nondimensional velocity units.
    """
    vstar = cr3bp_vstar_kms()
    return float(dv_kms) / max(vstar, 1e-12)


def minutes_to_nondim_time(minutes: float) -> float:
    """
    Convert physical time in minutes to CR3BP nondimensional time units.
    """
    seconds = 60.0 * float(minutes)
    return seconds / max(float(RUN.cr3bp_Tstar_s), 1e-12)


def nondim_time_to_minutes(t_nd: float) -> float:
    """
    Convert CR3BP nondimensional time units to physical minutes.
    """
    seconds = float(t_nd) * float(RUN.cr3bp_Tstar_s)
    return seconds / 60.0

def global_burn_cap_nondim() -> float:
    """
    Returns the global burn cap in nondimensional units.
    Uses RUN.global_burn_cap_kms and converts to nondimensional velocity.
    """
    return float(kms_to_nondim_dv(RUN.global_burn_cap_kms))

def tli_ballistic_trigger_nondim() -> float:
    """
    DV threshold above which a burn is considered a committed TLI.
    Returned in CR3BP nondimensional velocity units.
    """
    return float(kms_to_nondim_dv(RUN.tli_ballistic_trigger_kms))


def tli_departure_trigger_radius() -> float:
    """
    Earth-centered radius threshold above which the spacecraft is considered
    to have committed to departure, even if the last burn was below the
    ballistic-DV trigger threshold.
    """
    return float(RUN.tli_departure_trigger_rE)


def in_fine_integration_region(cfg: "CR3BPConfig", state: np.ndarray) -> bool:
    """
    Return True if the spacecraft is close enough to Earth or Moon that we want
    fine RK4 substeps for stability and impact detection robustness.
    """
    mu = float(cfg.mu)
    rE, rM = dist_to_primaries(mu, np.asarray(state, dtype=np.float64))
    thresh = float(RUN.fine_substep_region_radius)
    return (rE <= thresh) or (rM <= thresh)


def rk4_target_substep_nondim(dt_total_nd: float) -> float:
    """
    Adaptive RK4 target substep size for V3_2.

    Goal:
    - If the requested drift is very short, use about 1 minute per RK4 step.
    - If the requested drift is long, allow coarser RK4 steps up to a max target.
    - This keeps tau=-1 cheap without tanking performance for tau=+1.
    """
    dt_total_min = nondim_time_to_minutes(float(dt_total_nd))

    x0 = float(RUN.rk4_target_transition_min_minutes)
    x1 = float(RUN.rk4_target_transition_max_minutes)

    y0 = float(RUN.rk4_substep_target_min_minutes)
    y1 = float(RUN.rk4_substep_target_max_minutes)

    if x1 <= x0:
        x1 = x0 + 1.0

    if dt_total_min <= x0:
        target_min = y0
    elif dt_total_min >= x1:
        target_min = y1
    else:
        alpha = (dt_total_min - x0) / (x1 - x0)
        target_min = y0 + alpha * (y1 - y0)

    target_min = max(1e-6, target_min)
    return float(minutes_to_nondim_time(target_min))

def fine_rk4_substep_nondim() -> float:
    """
    Fine RK4 target substep used near Earth / Moon.
    """
    return float(minutes_to_nondim_time(RUN.fine_rk4_substep_minutes))


def apply_stage_to_cfg(base_cfg: CR3BPConfig, stage: CurriculumStage) -> CR3BPConfig:
    cfg = CR3BPConfig(**vars(base_cfg))

    cfg.trainer_mode = str(stage.trainer_mode)
    cfg.tli_control_mode = str(stage.tli_control_mode)

    cfg.mcc_enabled = bool(stage.mcc_enabled)
    cfg.tli_only_mode = bool(stage.tli_only_mode)
    cfg.reward_after_tli_ballistic_enabled = bool(stage.reward_after_tli_ballistic_enabled)

    cfg.spawn_theta_limit_enabled = bool(stage.spawn_theta_limit_enabled)
    cfg.spawn_theta_min = float(stage.spawn_theta_min)
    cfg.spawn_theta_max = float(stage.spawn_theta_max)

    cfg.ppo_b_baseline_theta = float(stage.ppo_b_baseline_theta)
    cfg.ppo_b_baseline_ax = float(stage.ppo_b_baseline_ax)
    cfg.ppo_b_baseline_ay = float(stage.ppo_b_baseline_ay)
    cfg.ppo_b_baseline_tau = float(stage.ppo_b_baseline_tau)
    cfg.ppo_b_baseline_state_noise_pos = float(stage.ppo_b_baseline_state_noise_pos)
    cfg.ppo_b_baseline_state_noise_vel = float(stage.ppo_b_baseline_state_noise_vel)

    cfg.ppo_b_case_source = str(stage.ppo_b_case_source)
    cfg.ppo_b_library_path = str(stage.ppo_b_library_path)

    cfg.ppo_b_prob_good = float(stage.ppo_b_prob_good)
    cfg.ppo_b_prob_savable = float(stage.ppo_b_prob_savable)
    cfg.ppo_b_prob_bad = float(stage.ppo_b_prob_bad)

    cfg.ppo_b_eval_use_same_distribution = bool(stage.ppo_b_eval_use_same_distribution)

    cfg.ppo_b_noise_theta_deg = float(stage.ppo_b_noise_theta_deg)
    cfg.ppo_b_noise_tli_dir_deg = float(stage.ppo_b_noise_tli_dir_deg)
    cfg.ppo_b_noise_tli_dv_kms = float(stage.ppo_b_noise_tli_dv_kms)

    cfg.ppo_b_use_fixed_index = bool(stage.ppo_b_use_fixed_index)
    cfg.ppo_b_fixed_index = int(stage.ppo_b_fixed_index)
    cfg.ppo_b_fixed_state_noise_pos = float(stage.ppo_b_fixed_state_noise_pos)
    cfg.ppo_b_fixed_state_noise_vel = float(stage.ppo_b_fixed_state_noise_vel)

    cfg.dv_noise_sigma_tli = float(stage.dv_noise_sigma_tli)
    cfg.dv_noise_sigma_mcc = float(stage.dv_noise_sigma_mcc)

    cfg.staged_tli_enabled = bool(stage.staged_tli_enabled)
    cfg.staged_tli_commit_on_cumulative_dv = bool(stage.staged_tli_commit_on_cumulative_dv)
    cfg.staged_tli_limit_burn_count = bool(stage.staged_tli_limit_burn_count)
    cfg.staged_tli_max_burn_count = int(stage.staged_tli_max_burn_count)
    cfg.staged_tli_min_commit_frac_of_target = float(stage.staged_tli_min_commit_frac_of_target)

    if getattr(stage, "staged_tli_cumulative_dv_target", None) is not None:
        cfg.staged_tli_cumulative_dv_target = float(stage.staged_tli_cumulative_dv_target)

    if RUN.tli_dv_max_kms is not None:
        cfg.dv_max_tli = kms_to_nondim_dv(float(RUN.tli_dv_max_kms))

    if RUN.mcc_dv_max_kms is not None:
        cfg.dv_max_mcc = kms_to_nondim_dv(float(RUN.mcc_dv_max_kms))

    return cfg




class SeanStyleReward:
    def __init__(self, config: RewardConfig, weights: RewardWeights):
        self.cfg = config
        self.w = weights
        self.min_rM = np.inf
        self.v_at_min_rM = 0.0

    def reset_episode(self):
        self.min_rM = np.inf
        self.v_at_min_rM = 0.0

    def update_closest_approach(self, rM, v_rel):
        if rM < self.min_rM:
            self.min_rM = rM
            self.v_at_min_rM = v_rel

    def dv_penalty(self, dv_step):
        return self.w.w_dv * (-dv_step / self.cfg.dv_scale)

    def dv_budget_terminal_penalty(self, dv_total):
        if dv_total > self.cfg.dv_budget:
            return -float(self.w.w_budget)
        return 0.0

    def escape_penalty(self, term_reason: str):
        if term_reason == "escape":
            return -float(self.w.w_escape)
        return 0.0
    
    def invalid_preflyby_earth_return_penalty(self, term_reason: str):
        if term_reason == "invalid_preflyby_earth_return":
            return -float(self.w.w_invalid_preflyby_earth_return)
        return 0.0

    def crash_penalty(self, env, rE, rM, info=None):
        if info is None:
            info = {}

        earth_r = float(self.cfg.earth_radius)
        moon_r = float(self.cfg.moon_radius)

        if env is not None and hasattr(env, "cfg"):
            earth_r = float(getattr(env.cfg, "r_earth_impact", earth_r))
            moon_r = float(getattr(env.cfg, "r_moon_impact", moon_r))

        # Earth crash
        if rE <= earth_r:
            flyby_done = bool(info.get("flyby_done", False))
            corridor_hit = bool(info.get("return_corridor_hit_postflyby", False))

            # Softer Earth crash only if THIS evaluated trajectory already completed return geometry
            if flyby_done and corridor_hit:
                return -float(self.w.w_postflyby_earth_crash)

            return -float(self.w.w_earth_crash)

        # Moon crash remains harsh
        if rM <= moon_r:
            return -float(self.w.w_moon_crash)

        return 0.0

    def flyby_distance_reward(self, min_rM_value: float, r_flyby: float = None):
        """
        Flyby reward based on env-tracked closest lunar approach.
        This should use substep-accurate geometry from env/info, not reward-model internal state.
        """
        rmin = float(min_rM_value)
        if not np.isfinite(rmin):
            return 0.0

        d0 = float(self.cfg.r0_distance_flyby)
        beta = float(self.cfg.beta_distance_flyby)

        if r_flyby is None:
            x = np.clip(rmin / max(d0, 1e-12), 0.0, 100.0)
            rd = 1.0 / (1.0 + x ** beta)
            return float(self.w.w_flyby) * float(rd)

        rf = float(r_flyby)
        d_eff = max(rmin, rf)

        x = np.clip(d_eff / max(d0, 1e-12), 0.0, 100.0)
        rd = 1.0 / (1.0 + x ** beta)

        x_rf = np.clip(rf / max(d0, 1e-12), 0.0, 100.0)
        rd_rf = 1.0 / (1.0 + x_rf ** beta)
        rd_norm = rd / max(rd_rf, 1e-12)

        return float(self.w.w_flyby) * float(rd_norm)


    def velocity_mismatch_reward(self, vrel_at_min_rM_value: float):
        """
        Velocity mismatch reward based on env-tracked relative speed at closest lunar approach.
        """
        vrel_min = float(vrel_at_min_rM_value)
        if not np.isfinite(vrel_min):
            return 0.0

        dv = abs(vrel_min - self.cfg.v_target_moon)
        dv_eff = max(0.0, dv - self.cfg.v_deadzone)
        rv = -dv_eff / self.cfg.dv_scale
        return float(self.w.w_velocity) * float(rv)
    
    def return_corridor_distance_from_actual_closest_approach(self, rE_postflyby: float, rp_min: float, rp_max: float):
        """
        Distance from actual post-flyby closest Earth approach to the allowed return corridor.
        Returns 0 if inside corridor.
        """
        if not np.isfinite(rE_postflyby):
            return np.inf

        if rE_postflyby < rp_min:
            return float(rp_min - rE_postflyby)

        if rE_postflyby > rp_max:
            return float(rE_postflyby - rp_max)

        return 0.0

    def return_reward(self, rE_postflyby: float, rp_min: float, rp_max: float):
        """
        Return shaping based on actual closest Earth distance after flyby.

        - If closest post-flyby Earth approach is inside [rp_min, rp_max],
        reward saturates at full w_return.
        - If outside, reward decays smoothly with distance to corridor.
        """
        if not np.isfinite(rE_postflyby):
            return 0.0

        d = self.return_corridor_distance_from_actual_closest_approach(rE_postflyby, rp_min, rp_max)

        beta = float(self.cfg.beta_distance_return)
        d0 = float(self.cfg.r0_distance_return)

        x = np.clip(d / max(d0, 1e-12), 0.0, 100.0)
        rr = 1.0 / (1.0 + x ** beta)

        return float(self.w.w_return) * float(rr)

    def compute(
        self,
        env=None,
        state=None,
        info=None,
        dv_mag=0.0,
        terminated=False,
        truncated=False,
        **kwargs
    ):
        if "trunc" in kwargs and truncated is False:
            truncated = bool(kwargs["trunc"])
        if info is None and "event_info" in kwargs:
            info = kwargs["event_info"]
        if info is None:
            info = {}

        if state is None:
            if env is not None and hasattr(env, "state"):
                state = env.state
            else:
                state = np.zeros(4)

        rM = float(info.get("rM", np.inf))
        rE = float(info.get("rE", np.inf))
        vrel = float(info.get("vrel_moon", 0.0))
        dv_total = float(info.get("dv_used", 0.0))

        # Use env-tracked substep quantities as source of truth
        min_rM_env = float(info.get("min_rM", np.inf))
        vrel_at_min_rM_env = float(info.get("vrel_at_min_rM", np.nan))
        rE_postflyby = float(info.get("min_rE_postflyby", np.inf))

        if not np.isfinite(rM):
            rM = 10.0
        if not np.isfinite(rE):
            rE = 10.0
        if not np.isfinite(vrel):
            vrel = 0.0
        if not np.isfinite(dv_total):
            dv_total = 0.0

        if not np.isfinite(min_rM_env):
            min_rM_env = np.inf
        if not np.isfinite(vrel_at_min_rM_env):
            vrel_at_min_rM_env = np.nan
        if not np.isfinite(rE_postflyby):
            rE_postflyby = np.inf

        # Keep these only as debug mirrors so reports still show something sensible
        self.min_rM = float(min_rM_env) if np.isfinite(min_rM_env) else np.inf
        self.v_at_min_rM = float(vrel_at_min_rM_env) if np.isfinite(vrel_at_min_rM_env) else 0.0

        reward = 0.0
        terms: Dict[str, float] = {}


        # -------------------------------------------------
        # Per-step penalties
        # -------------------------------------------------
        r_dv = self.dv_penalty(float(dv_mag))
        reward += r_dv
        terms["r_dv"] = r_dv

        # Budget penalty is now TERMINAL ONLY
        r_budget = 0.0
        terms["r_budget"] = r_budget

        term_reason = str(info.get("term_reason", ""))

        r_escape = self.escape_penalty(term_reason)
        reward += r_escape
        terms["r_escape"] = r_escape

        r_invalid_preflyby = self.invalid_preflyby_earth_return_penalty(term_reason)
        reward += r_invalid_preflyby
        terms["r_invalid_preflyby_earth_return"] = r_invalid_preflyby

        r_crash = self.crash_penalty(env, rE, rM, info=info)
        reward += r_crash
        terms["r_crash"] = r_crash

        
        # -------------------------------------------------
        # Terminal rewards
        # -------------------------------------------------
        if terminated or truncated:
            left_leo = bool(info.get("left_leo", False))
            flyby_done = bool(info.get("flyby_done", False))
            term_reason = str(info.get("term_reason", ""))
            bootstrap_pre_tli_timeout = bool(
                getattr(env.cfg, "tli_only_mode", False)
                and term_reason == "no_tli_3_orbits"
            )

            crashed = term_reason in ("moon_impact", "earth_impact")
            escaped = term_reason == "escape"
            timed_out = bool(truncated) or (term_reason == "timeout")

            corridor_hit_postflyby = bool(info.get("return_corridor_hit_postflyby", False))

            # Suppress positive terminal rewards on:
            # - Moon crash always
            # - Earth crash only if corridor was NOT already hit post-flyby
            suppress_positive_terminal_rewards = (
                (term_reason == "moon_impact") or
                (term_reason == "earth_impact" and not corridor_hit_postflyby) or
                (term_reason == "dv_budget_exceeded")
            )

            # ----------------------------
            # One-time terminal budget penalty
            # ----------------------------
            r_budget = self.dv_budget_terminal_penalty(dv_total)
            reward += r_budget
            terms["r_budget"] = r_budget

            # ----------------------------
            # Flyby reward
            # ----------------------------
            did_meaningful_moon_approach = bool(
                np.isfinite(self.min_rM) and self.min_rM <= self.cfg.flyby_reward_gate
            )

            allow_flyby_reward = bool(
                (not suppress_positive_terminal_rewards)
                and (flyby_done or did_meaningful_moon_approach)
                and (
                    left_leo
                    or bootstrap_pre_tli_timeout
                )
            )

            if allow_flyby_reward:
                r_flyby = self.flyby_distance_reward(
                    min_rM_value=min_rM_env,
                    r_flyby=float(getattr(env.cfg, "r_moon_flyby", self.cfg.moon_radius)) if env is not None and hasattr(env, "cfg") else None,
                )

                r_vel = self.velocity_mismatch_reward(
                    vrel_at_min_rM_value=vrel_at_min_rM_env
                )
            else:
                r_flyby = 0.0
                r_vel = 0.0

            reward += r_flyby
            reward += r_vel
            terms["r_flyby"] = r_flyby
            terms["r_velocity"] = r_vel

            # ----------------------------
            # Return reward
            # ----------------------------
            # Allow return shaping only after flyby, and never on crash/escape.
            # Keep timeout eligible if you want "near return" trajectories to still get signal.
            corridor_hit_postflyby = bool(info.get("return_corridor_hit_postflyby", False))

            if (not suppress_positive_terminal_rewards) and flyby_done:
                rp_min = getattr(env.cfg, "rp_min", None) if env is not None and hasattr(env, "cfg") else None
                rp_max = getattr(env.cfg, "rp_max", None) if env is not None and hasattr(env, "cfg") else None

                if (
                    rp_min is not None and rp_max is not None
                    and np.isfinite(rp_min) and np.isfinite(rp_max)
                    and (rp_max > rp_min)
                ):
                    # If the trajectory ever hit the corridor post-flyby, saturate to full return reward
                    if corridor_hit_postflyby:
                        r_return = float(self.w.w_return)
                    else:
                        r_return = self.return_reward(rE_postflyby, float(rp_min), float(rp_max))
                else:
                    r_return = 0.0
            else:
                r_return = 0.0

            reward += r_return
            terms["r_return"] = r_return

            # ----------------------------
            # Debug flags
            # ----------------------------
            if suppress_positive_terminal_rewards:
                terms["terminal_rewards_suppressed"] = 1.0
                if crashed:
                    terms["terminal_suppression_reason_crash"] = 1.0
            elif escaped:
                terms["terminal_escape"] = 1.0
            elif timed_out and term_reason != "success":
                terms["terminal_timeout"] = 1.0
            elif term_reason == "success":
                terms["terminal_success"] = 1.0

            terms["terminal_left_leo"] = 1.0 if left_leo else 0.0
            terms["terminal_bootstrap_pre_tli_timeout"] = 1.0 if bootstrap_pre_tli_timeout else 0.0
            terms["terminal_meaningful_moon_approach"] = 1.0 if did_meaningful_moon_approach else 0.0
            terms["terminal_flyby_reward_allowed"] = 1.0 if allow_flyby_reward else 0.0
            terms["terminal_return_eligible"] = 1.0 if ((not suppress_positive_terminal_rewards) and flyby_done) else 0.0
            terms["terminal_corridor_hit_postflyby"] = 1.0 if corridor_hit_postflyby else 0.0
            terms["debug_env_min_rM"] = float(min_rM_env) if np.isfinite(min_rM_env) else 0.0
            terms["debug_env_vrel_at_min_rM"] = float(vrel_at_min_rM_env) if np.isfinite(vrel_at_min_rM_env) else 0.0
            terms["debug_env_min_rE_postflyby"] = float(rE_postflyby) if np.isfinite(rE_postflyby) else 0.0

        if not np.isfinite(reward):
            reward = 0.0

        for k, v in list(terms.items()):
            if isinstance(v, (float, int, np.floating)) and (not np.isfinite(v)):
                terms[k] = 0.0


        return float(reward), terms




# ============================================================
# 3) CR3BP DYNAMICS + GEOMETRY HELPERS
# ============================================================

def cr3bp_planar_deriv(mu: float, state: np.ndarray) -> np.ndarray:
    x, y, vx, vy = state

    r1 = math.sqrt((x + mu) ** 2 + y ** 2)
    r2 = math.sqrt((x - 1 + mu) ** 2 + y ** 2)
    r1 = max(r1, 1e-9)
    r2 = max(r2, 1e-9)

    dUdx = x - (1 - mu) * (x + mu) / (r1 ** 3) - mu * (x - 1 + mu) / (r2 ** 3)
    dUdy = y - (1 - mu) * y / (r1 ** 3) - mu * y / (r2 ** 3)

    ax = 2 * vy + dUdx
    ay = -2 * vx + dUdy
    return np.array([vx, vy, ax, ay], dtype=np.float64)

def rk4_step(mu: float, state: np.ndarray, dt: float) -> np.ndarray:
    k1 = cr3bp_planar_deriv(mu, state)
    k2 = cr3bp_planar_deriv(mu, state + 0.5 * dt * k1)
    k3 = cr3bp_planar_deriv(mu, state + 0.5 * dt * k2)
    k4 = cr3bp_planar_deriv(mu, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def jacobi_constant(mu: float, state: np.ndarray) -> float:
    x, y, vx, vy = state
    r1 = math.sqrt((x + mu) ** 2 + y ** 2)
    r2 = math.sqrt((x - 1 + mu) ** 2 + y ** 2)
    r1 = max(r1, 1e-9)
    r2 = max(r2, 1e-9)
    Omega = 0.5 * (x*x + y*y) + (1 - mu) / r1 + mu / r2
    C = 2 * Omega - (vx*vx + vy*vy)
    return float(C)

def earth_moon_positions(mu: float) -> Tuple[np.ndarray, np.ndarray]:
    rE = np.array([-mu, 0.0], dtype=np.float64)
    rM = np.array([1 - mu, 0.0], dtype=np.float64)
    return rE, rM

def dist_to_primaries(mu: float, state: np.ndarray) -> Tuple[float, float]:
    x, y, _, _ = state
    rE, rM = earth_moon_positions(mu)
    r_e = float(np.hypot(x - rE[0], y - rE[1]))
    r_m = float(np.hypot(x - rM[0], y - rM[1]))
    return r_e, r_m

def dist_to_interval(x: float, xmin: float, xmax: float) -> float:
    if x < xmin:
        return float(xmin - x)
    if x > xmax:
        return float(x - xmax)
    return 0.0

def radial_velocity_about_point(r: np.ndarray, v: np.ndarray) -> float:
    rn = np.linalg.norm(r)
    if rn < 1e-12:
        return 0.0
    return float(np.dot(r, v) / rn)

def omega_cross_r_2d(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(2,)
    return np.array([-r[1], r[0]], dtype=np.float64)

def v_rot_to_inertial(pos: np.ndarray, v_rot: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64).reshape(2,)
    v_rot = np.asarray(v_rot, dtype=np.float64).reshape(2,)
    return v_rot + omega_cross_r_2d(pos)



def wrap_pi(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))

def phase_angle_sc_vs_moon_about_earth(mu: float, state: np.ndarray) -> float:
    """
    Earth-centered spacecraft phase angle relative to the Moon line, wrapped to [-pi, pi].

    This function defines the underlying geometric phase used by tau.
    A true tau value of 0 corresponds to alignment with the Earth-to-Moon line.
    """
    rE_pos, rM_pos = earth_moon_positions(mu)
    pos = np.asarray(state[:2], dtype=np.float64)
    r_sc_E = pos - rE_pos
    r_m_E = rM_pos - rE_pos

    theta_sc = math.atan2(float(r_sc_E[1]), float(r_sc_E[0]))
    theta_m  = math.atan2(float(r_m_E[1]),  float(r_m_E[0]))
    return wrap_pi(theta_sc - theta_m)

def round_to_nearest_rollout_multiple(x: int) -> int:
    block = ppo_rollout_block_size()
    return int(round(float(x) / float(block)) * block)


def snap_curriculum_timesteps(curriculum):
    block = ppo_rollout_block_size()

    print("\nCurriculum timestep snapping")
    print("-" * 60)
    print(f"PPO rollout block = {block}")
    print("")

    for stage in curriculum:
        original = int(stage.timesteps)
        snapped = round_to_nearest_rollout_multiple(original)

        stage.timesteps = snapped

        print(
            f"{stage.name:>20} : "
            f"{original:>12,}  ->  {snapped:>12,}"
        )

    print("-" * 60)


def build_reward_factory(reward_cfg: RewardConfig, weights: RewardWeights):
    def _factory():
        return SeanStyleReward(copy.deepcopy(reward_cfg), weights)
    return _factory


def get_obs_schema(env) -> List[str]:
    """
    Return the exact observation-field names corresponding to env._get_obs().

    This MUST mirror the logic in _get_obs() so exported observation vectors
    can always be interpreted correctly later, even if observation toggles
    change between experiments.
    """
    names: List[str] = [
        "x_scaled",
        "y_scaled",
        "vx_scaled",
        "vy_scaled",
        "rE_scaled",
        "rM_scaled",
        "jacobi_scaled",
        "t_over_tmax",
        "dv_used_over_budget",
    ]

    if bool(env.cfg.add_phase_angle_obs):
        names.append("phase_sc_vs_moon_about_earth_over_pi")

    if bool(env.cfg.add_mode_obs) and bool(env.cfg.add_legacy_mode_obs):
        names.extend([
            "tli_used_flag",
            "tau_max_now_over_tau_max_global",
            "dv_cap_now_over_tli_cap_ref",
            "pre_tli_clock",
        ])

    if (
        bool(env.cfg.add_mode_obs)
        and bool(env.cfg.add_staged_tli_obs)
        and bool(env.cfg.staged_tli_enabled)
    ):
        names.extend([
            "pre_tli_cum_dv_over_target",
            "pre_tli_burn_count_over_max",
        ])

    return names

# ============================================================
# 5) THE ENVIRONMENT
# CR3BP free-return environment with ballistic tau time-warp.
# ============================================================



class CR3BPFreeReturnEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg: CR3BPConfig, seed: Optional[int] = None, reward_model=None):
        super().__init__()
        self.cfg = cfg
        self.reward_model = reward_model
        self.rng = np.random.default_rng(seed)

        self.action_history: List[Dict[str, Any]] = []

        if self._use_tangential_scalar_action():
            # action = [signed_tangential_dv_raw, tau_raw]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            # action = [ax, ay, tau_raw]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        base_dim = 9
        phase_dim = 1 if self.cfg.add_phase_angle_obs else 0
        legacy_mode_dim = 4 if (self.cfg.add_mode_obs and self.cfg.add_legacy_mode_obs) else 0
        staged_tli_dim = 2 if (
            self.cfg.add_mode_obs
            and self.cfg.add_staged_tli_obs
            and self.cfg.staged_tli_enabled
        ) else 0
        obs_dim = base_dim + phase_dim + legacy_mode_dim + staged_tli_dim

        

        high = np.array([np.finfo(np.float32).max] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.debug_eval = False

        # state
        self.state = np.zeros(4, dtype=np.float64)
        self.t = 0.0
        self.dv_used = 0.0

        self.global_step = 0

        # events
        self.flyby_done = False
        self.return_done = False
        self.success = False

        self._early_terminate = None
        self._substep_events = {
            "flyby": False,
            "corridor": False,
            "corridor_exit_outward": False,
        }

        # logs
        self.traj: List[np.ndarray] = []
        self.t_hist: List[float] = []
        self.burns: List[np.ndarray] = []
        self.info_last: Dict[str, Any] = {}

        # V3 logging
        self.last_dt_effective = 0.0
        self.last_dt_warp = 0.0
        self.last_dt_post = 0.0
        self.last_tli_u01_raw = np.nan
        self.last_tli_u01_exec = np.nan

        self.ballistic_tli_reward_last = 0.0
        self.ballistic_tli_min_rM_last = np.nan
        self.ballistic_tli_min_rE_postflyby_last = np.nan
        self.ballistic_tli_corridor_dist_last = np.nan
        self.ballistic_tli_corridor_hit_last = False
        self.ballistic_tli_success_last = False
        self.ballistic_tli_vrel_at_min_rM_last = np.nan
        self.ballistic_terminal_marker_rot = None
        self.terminal_marker_rot = None
        self.mcc_ballistic_overlays: List[Dict[str, Any]] = []
        self.spawn_theta = float("nan")
        self.tli_theta = float("nan")
        self.tli_pos_rot = None
        # PPO-B scenario library cache
        self.ppo_b_library_loaded = False
        self.ppo_b_library_cache: Dict[str, np.ndarray] = {}

    
    def _use_tangential_scalar_action(self) -> bool:
        """
        Use 2D action space [signed_tangential_dv_raw, tau_raw]
        only for PPO-A tangential-control runs.

        PPO-B should keep the normal 3D action space [ax, ay, tau_raw].
        """
        trainer_mode = str(getattr(self.cfg, "trainer_mode", "ppo_a")).lower()
        control_mode = str(getattr(self.cfg, "tli_control_mode", "full")).lower()

        return (trainer_mode == "ppo_a") and (control_mode == "tangential")
    

    def _dv_vec_from_tangential_scalar(self, u_raw: float, dv_cap: float) -> np.ndarray:
        """
        Map one scalar action in [-1, 1] to a signed tangential burn:
            u_raw = -1  -> max retrograde tangential burn
            u_raw =  0  -> zero burn
            u_raw = +1  -> max prograde tangential burn
        """
        u_raw = float(np.clip(u_raw, -1.0, 1.0))
        t_hat = self._local_tangential_hat_rot_about_earth(self.state)
        signed_mag = float(dv_cap) * u_raw
        return signed_mag * t_hat


    def set_global_step(self, global_step: int):
        self.global_step = int(global_step)

    def set_debug_eval(self, flag: bool):
        self.debug_eval = bool(flag)
    
    def _build_leo_state_from_theta(self, theta: float) -> np.ndarray:
        mu = float(self.cfg.mu)
        rE_pos, _ = earth_moon_positions(mu)

        r0 = float(self.cfg.r0_earth)
        x = rE_pos[0] + r0 * np.cos(theta)
        y = rE_pos[1] + r0 * np.sin(theta)

        r_rel = np.array([x - rE_pos[0], y - rE_pos[1]], dtype=np.float64)

        muE = 1.0 - mu
        v_circ = math.sqrt(muE / max(r0, 1e-12))

        t_hat = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
        v_rel_I = v_circ * t_hat

        omega_x_r = omega_cross_r_2d(r_rel)
        v_rot = v_rel_I - omega_x_r

        vx, vy = float(v_rot[0]), float(v_rot[1])
        return np.array([x, y, vx, vy], dtype=np.float64)
    

    def _earth_centered_theta_of_state(self, state: np.ndarray) -> float:
        mu = float(self.cfg.mu)
        rE_pos, _ = earth_moon_positions(mu)
        pos = np.asarray(state[:2], dtype=np.float64)
        r_sc_E = pos - rE_pos
        return float(np.arctan2(r_sc_E[1], r_sc_E[0]))
    
    def _local_tangential_hat_rot_about_earth(self, state: np.ndarray) -> np.ndarray:
        mu = float(self.cfg.mu)
        rE_pos, _ = earth_moon_positions(mu)

        pos = np.asarray(state[:2], dtype=np.float64)
        r_sc_E = pos - rE_pos
        r_norm = float(np.linalg.norm(r_sc_E))

        if r_norm < 1e-12:
            return np.array([1.0, 0.0], dtype=np.float64)

        r_hat = r_sc_E / r_norm
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
        return t_hat
    
    def _apply_pre_tli_control_mode(self, dv_cmd_nominal: np.ndarray) -> np.ndarray:
        dv_cmd_nominal = np.asarray(dv_cmd_nominal, dtype=np.float64).reshape(2,)

        mode = str(getattr(self.cfg, "tli_control_mode", "full")).lower()

        if mode != "tangential":
            return dv_cmd_nominal.copy()

        t_hat = self._local_tangential_hat_rot_about_earth(self.state)
        signed_mag = float(np.dot(dv_cmd_nominal, t_hat))
        return signed_mag * t_hat
    
    def _apply_post_tli_spawn_noise(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).copy()

        pos_sigma = float(getattr(self.cfg, "ppo_b_baseline_state_noise_pos", 0.0))
        vel_sigma = float(getattr(self.cfg, "ppo_b_baseline_state_noise_vel", 0.0))

        if pos_sigma > 0.0:
            s[:2] += self.rng.normal(0.0, pos_sigma, size=2)

        if vel_sigma > 0.0:
            s[2:4] += self.rng.normal(0.0, vel_sigma, size=2)

        return s
    
    def _build_post_tli_state_from_baseline(self) -> Dict[str, Any]:
        theta = float(self.cfg.ppo_b_baseline_theta)
        ax_raw = float(self.cfg.ppo_b_baseline_ax)
        ay_raw = float(self.cfg.ppo_b_baseline_ay)
        tau_raw = float(self.cfg.ppo_b_baseline_tau)

        state0 = self._build_leo_state_from_theta(theta)

        # Temporarily use that state so tangential projection works correctly
        old_state = self.state.copy()
        self.state = state0.copy()

        if self._use_tangential_scalar_action():
            t_hat = self._local_tangential_hat_rot_about_earth(self.state)
            dv_vec_xy = self._dv_vec_from_action_xy(
                ax_raw,
                ay_raw,
                dv_cap=self._dv_cap_tli(),
            )
            signed_u = float(np.dot(dv_vec_xy, t_hat) / max(self._dv_cap_tli(), 1e-12))
            signed_u = float(np.clip(signed_u, -1.0, 1.0))
            dv_cmd_nominal = self._dv_vec_from_tangential_scalar(
                signed_u,
                dv_cap=self._dv_cap_tli(),
            )
        else:
            dv_cmd_nominal = self._dv_vec_from_action_xy(
                ax_raw,
                ay_raw,
                dv_cap=self._dv_cap_tli(),
            )
            dv_cmd_nominal = self._apply_pre_tli_control_mode(dv_cmd_nominal)
        dv_mag = float(np.linalg.norm(dv_cmd_nominal))

        state1 = state0.copy()
        state1[2] += dv_cmd_nominal[0]
        state1[3] += dv_cmd_nominal[1]

        dt_post = self._drift_time_from_tau_masked(tau_raw)
        state2 = self._propagate_copy(state1, dt_post)

        self.state = old_state

        return {
            "state_post_tli": state2,
            "state_immediate_after_burn": state1,
            "dv_vec_rot": dv_cmd_nominal.copy(),
            "dv_mag": float(dv_mag),
            "tau_raw": float(tau_raw),
            "ax_raw": float(ax_raw),
            "ay_raw": float(ay_raw),
            "theta": float(theta),
            "dt_post": float(dt_post),
        }
    
    def _set_post_tli_initial_condition(self, payload: Dict[str, Any]) -> None:


        state_post_tli = np.asarray(payload["state_post_tli"], dtype=np.float64).copy()

        # --------------------------------------------------------
        # Apply old baseline post-TLI spawn noise ONLY for baseline mode
        # Do NOT apply it for handoff-library fixed-case mode
        # --------------------------------------------------------
        trainer_mode = str(getattr(self.cfg, "trainer_mode", "ppo_a")).lower()
        case_source = str(getattr(self.cfg, "ppo_b_case_source", "")).lower()

        use_baseline_spawn_noise = (
            trainer_mode == "ppo_b_baseline"
            or (trainer_mode == "ppo_b_library" and case_source == "baseline")
        )

        if use_baseline_spawn_noise:
            state_post_tli = self._apply_post_tli_spawn_noise(state_post_tli)

        self.state = state_post_tli
        self.t = float(payload.get("dt_post", 0.0))

        

        self.tli_state_after_burn = np.asarray(
            payload.get("state_immediate_after_burn", state_post_tli),
            dtype=np.float64
        ).copy()

        self.tli_theta = self._earth_centered_theta_of_state(self.tli_state_after_burn)
        self.tli_pos_rot = self.tli_state_after_burn[:2].copy()

        self.tli_used = True
        self.tli_executed = True
        self.tli_ballistic_reward_given = True

        self.dv0 = float(payload.get("dv_mag", 0.0))
        self.dv_used = float(payload.get("dv_mag", 0.0))
        self.dv_mcc_total = 0.0

        self.tli_ax = float(payload.get("ax_raw", np.nan))
        self.tli_ay = float(payload.get("ay_raw", np.nan))
        self.tli_tau = float(payload.get("tau_raw", np.nan))
        self.tli_step_executed = -1

        self.last_dt_effective = float(payload.get("dt_post", 0.0))
        self.last_dt_warp = 0.0
        self.last_dt_post = float(payload.get("dt_post", 0.0))

        self.ballistic_tli_reward_last = 0.0
        self.ballistic_tli_min_rM_last = np.nan
        self.ballistic_tli_min_rE_postflyby_last = np.nan
        self.ballistic_tli_corridor_dist_last = np.nan
        self.ballistic_tli_corridor_hit_last = False
        self.ballistic_tli_success_last = False
        self.ballistic_tli_vrel_at_min_rM_last = np.nan

        self.ppo_b_scenario_index = int(payload.get("scenario_index", -1))
        self.ppo_b_scenario_row_index = int(payload.get("scenario_row_index", -1))
        self.ppo_b_scenario_label = int(payload.get("scenario_label", -1))
        self.ppo_b_scenario_term_reason = str(payload.get("scenario_term_reason", ""))
        self.ppo_b_scenario_ballistic_min_rM = float(payload.get("scenario_ballistic_min_rM", np.nan))
        self.ppo_b_scenario_ballistic_corridor_dist = float(payload.get("scenario_ballistic_corridor_dist", np.nan))
        self.ppo_b_scenario_ballistic_success = bool(payload.get("scenario_ballistic_success", False))
        self.ppo_b_scenario_fixed_index_mode = bool(payload.get("scenario_fixed_index_mode", False))
        self.ppo_b_scenario_noise_pos_sigma = float(payload.get("scenario_noise_pos_sigma", 0.0))
        self.ppo_b_scenario_noise_vel_sigma = float(payload.get("scenario_noise_vel_sigma", 0.0))

        rE_now, _ = dist_to_primaries(self.cfg.mu, self.state)
        self.max_rE_seen_post_tli = float(rE_now)
        self.invalid_return_armed_episode = bool(
            rE_now >= float(self.cfg.ballistic_invalid_return_arm_rE)
        )

        if self.tli_state_after_burn is not None:
            self._build_ballistic_reference_from_tli()
        
    def _rotate_vec_2d(self, vec: np.ndarray, angle_rad: float) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64).reshape(2,)
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return R @ vec

    def _wrap_0_2pi(self, angle: float) -> float:
        return float(angle % (2.0 * np.pi))

    def _load_ppo_b_scenario_library(self) -> None:
        """
        Load the PPO-B scenario library from cfg.ppo_b_library_path.
        Required arrays depend on library type.

        Supported library formats:
        1) seed-command library:
            theta, ax_raw, ay_raw, tau_raw, label

        2) handoff-state library:
            state_handoff, label
            optional metadata arrays may also exist
        """
        if self.ppo_b_library_loaded:
            return

        raw_path = str(getattr(self.cfg, "ppo_b_library_path", "")).strip()
        if raw_path == "":
            raise ValueError(
                "trainer_mode='ppo_b_library' requires cfg.ppo_b_library_path "
                "to point to a .npz file."
            )

        path_obj = Path(raw_path)

        # If relative, resolve relative to THIS file's folder
        if not path_obj.is_absolute():
            path_obj = Path(__file__).resolve().parent / path_obj

        path_obj = path_obj.resolve()

        if not path_obj.exists():
            raise FileNotFoundError(
                f"PPO-B scenario library file not found:\n{path_obj}"
            )

        data = np.load(path_obj, allow_pickle=False)

        cache: Dict[str, np.ndarray] = {}

        # --------------------------------------------------------
        # Format A: seed-command library
        # --------------------------------------------------------
        if all(k in data for k in ["theta", "ax_raw", "ay_raw", "tau_raw", "label"]):
            theta = np.asarray(data["theta"], dtype=np.float64).reshape(-1)
            ax_raw = np.asarray(data["ax_raw"], dtype=np.float64).reshape(-1)
            ay_raw = np.asarray(data["ay_raw"], dtype=np.float64).reshape(-1)
            tau_raw = np.asarray(data["tau_raw"], dtype=np.float64).reshape(-1)
            label = np.asarray(data["label"], dtype=np.int32).reshape(-1)

            n = len(theta)
            if not (len(ax_raw) == len(ay_raw) == len(tau_raw) == len(label) == n):
                raise ValueError(
                    f"PPO-B scenario library arrays must all have same length. "
                    f"Got lengths: theta={len(theta)}, ax_raw={len(ax_raw)}, "
                    f"ay_raw={len(ay_raw)}, tau_raw={len(tau_raw)}, label={len(label)}"
                )

            cache["library_kind"] = np.array(["seed_commands"])
            cache["theta"] = theta
            cache["ax_raw"] = ax_raw
            cache["ay_raw"] = ay_raw
            cache["tau_raw"] = tau_raw
            cache["label"] = label

        # --------------------------------------------------------
        # Format B: handoff-state library
        # --------------------------------------------------------
        elif all(k in data for k in ["state_handoff", "label"]):
            state_handoff = np.asarray(data["state_handoff"], dtype=np.float64)
            label = np.asarray(data["label"], dtype=np.int32).reshape(-1)

            if state_handoff.ndim != 2 or state_handoff.shape[1] != 4:
                raise ValueError(
                    f"'state_handoff' must have shape (N,4), got {state_handoff.shape}"
                )

            n = state_handoff.shape[0]
            if len(label) != n:
                raise ValueError(
                    f"'state_handoff' and 'label' length mismatch: "
                    f"{n} vs {len(label)}"
                )

            cache["library_kind"] = np.array(["handoff_states"])
            cache["state_handoff"] = state_handoff
            cache["label"] = label

        else:
            raise ValueError(
                f"Unrecognized PPO-B library format in file:\n{path_obj}\n"
                "Expected either:\n"
                "  theta, ax_raw, ay_raw, tau_raw, label\n"
                "or:\n"
                "  state_handoff, label"
            )

        # Optional metadata
        optional_keys = [
            "dv_kms",
            "ballistic_min_rM",
            "ballistic_corridor_dist",
            "ballistic_success",
            "theta",
            "ax_raw",
            "ay_raw",
            "tau_raw",
            "state_tli",
            "t_handoff",
        ]
        for key in optional_keys:
            if key in data and key not in cache:
                cache[key] = np.asarray(data[key])

        self.ppo_b_library_cache = cache
        self.ppo_b_library_loaded = True

        print(f"[PPO-B LIBRARY] Loaded: {path_obj}")
        print(f"[PPO-B LIBRARY] Kind  : {cache['library_kind'][0]}")
        print(f"[PPO-B LIBRARY] Cases : {len(cache['label'])}")

    def _normalize_ppo_b_label_probs(self) -> Dict[int, float]:
        """
        Return normalized probabilities for labels:
            0 = good
            1 = savable
            2 = bad
        """
        probs = {
            0: max(0.0, float(getattr(self.cfg, "ppo_b_prob_good", 0.0))),
            1: max(0.0, float(getattr(self.cfg, "ppo_b_prob_savable", 0.0))),
            2: max(0.0, float(getattr(self.cfg, "ppo_b_prob_bad", 0.0))),
        }
        total = sum(probs.values())
        if total <= 0.0:
            probs = {0: 0.2, 1: 0.6, 2: 0.2}
            total = 1.0

        for k in probs:
            probs[k] /= total
        return probs
    
    def _build_post_tli_state_from_handoff_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build PPO-B initial condition directly from a stored post-TLI handoff state.
        Supports optional local state noise for fixed-index training.

        IMPORTANT:
        All payload fields must be finite. NaNs here can leak into the observation
        and crash PPO with NaN policy outputs.
        """
        state_handoff = np.asarray(case["state_handoff"], dtype=np.float64).copy()
        t_handoff = float(case["t_handoff"])

        # Use stored state immediately after TLI burn if available.
        # If missing, fall back to handoff state.
        state_after_tli = np.asarray(
            case.get("state_after_tli", state_handoff),
            dtype=np.float64
        ).copy()

        # --------------------------------------------------------
        # Optional local fixed-case noise
        # Applies only when using one exact library index
        # --------------------------------------------------------
        applied_pos_sigma = 0.0
        applied_vel_sigma = 0.0

        if bool(getattr(self.cfg, "ppo_b_use_fixed_index", False)):
            pos_sigma = float(getattr(self.cfg, "ppo_b_fixed_state_noise_pos", 0.0))
            vel_sigma = float(getattr(self.cfg, "ppo_b_fixed_state_noise_vel", 0.0))

            if pos_sigma > 0.0:
                state_handoff[0:2] += self.rng.normal(0.0, pos_sigma, size=2)
                applied_pos_sigma = pos_sigma

            if vel_sigma > 0.0:
                state_handoff[2:4] += self.rng.normal(0.0, vel_sigma, size=2)
                applied_vel_sigma = vel_sigma

        # --------------------------------------------------------
        # CRITICAL: keep these finite
        # --------------------------------------------------------
        theta_val = case.get("spawn_theta", case.get("theta", 0.0))
        try:
            theta_val = float(theta_val)
        except Exception:
            theta_val = 0.0
        if not np.isfinite(theta_val):
            theta_val = 0.0

        dv0_val = case.get("dv0", 0.0)
        try:
            dv0_val = float(dv0_val)
        except Exception:
            dv0_val = 0.0
        if not np.isfinite(dv0_val):
            dv0_val = 0.0

        payload = {
            "state_post_tli": state_handoff,
            "state_immediate_after_burn": state_after_tli,

            # Keep finite placeholders, never NaN
            "dv_vec_rot": np.zeros(2, dtype=np.float64),
            "dv_mag": dv0_val,
            "ax_raw": 0.0,
            "ay_raw": 0.0,
            "tau_raw": -1.0,

            "theta": theta_val,
            "dt_post": float(t_handoff),

            "scenario_index": int(case.get("index", -1)),
            "scenario_row_index": int(case.get("row_index", -1)),
            "scenario_label": int(case.get("label", -1)),
            "scenario_term_reason": str(case.get("term_reason", "")),
            "scenario_ballistic_min_rM": float(case.get("min_rM", np.nan)),
            "scenario_ballistic_corridor_dist": float(case.get("min_corridor_dist", np.nan)),
            "scenario_ballistic_success": bool(case.get("success", False)),

            # Noise bookkeeping
            "scenario_fixed_index_mode": bool(getattr(self.cfg, "ppo_b_use_fixed_index", False)),
            "scenario_noise_pos_sigma": float(applied_pos_sigma),
            "scenario_noise_vel_sigma": float(applied_vel_sigma),
        }
        return payload


    def _get_ppo_b_case_from_library_index(self, chosen_idx: int) -> Dict[str, Any]:
        """
        Return one exact PPO-B library case by index.
        Supports both seed-based and handoff-state libraries.
        """
        self._load_ppo_b_scenario_library()
        lib = self.ppo_b_library_cache

        labels = np.asarray(lib["label"], dtype=np.int32).reshape(-1)
        n_cases = len(labels)

        chosen_idx = int(chosen_idx)
        if chosen_idx < 0 or chosen_idx >= n_cases:
            raise IndexError(
                f"ppo_b_fixed_index={chosen_idx} is out of range for library size {n_cases}"
            )

        chosen_label = int(labels[chosen_idx])

        fmt = str(
            np.asarray(
                lib.get("library_kind", lib.get("library_format", ["seed_commands"]))
            ).reshape(-1)[0]
        )

        case = {
            "index": chosen_idx,
            "label": chosen_label,
            "library_format": fmt,
        }

        if fmt == "handoff_states":
            case["state_handoff"] = np.asarray(lib["state_handoff"][chosen_idx], dtype=np.float64).copy()
            case["t_handoff"] = float(lib["t_handoff"][chosen_idx])

            for key in [
                "theta",
                "dv_kms",
                "term_reason",
                "success",
                "min_rM",
                "min_corridor_dist",
                "row_index",
                "state_after_tli",
                "t_after_tli",
                "rE_after_tli",
                "rM_after_tli",
                "dv0",
                "tli_theta",
                "spawn_theta",
            ]:
                if key in lib and len(lib[key]) > chosen_idx:
                    case[key] = lib[key][chosen_idx]
        else:
            case["theta"] = float(lib["theta"][chosen_idx])
            case["ax_raw"] = float(lib["ax_raw"][chosen_idx])
            case["ay_raw"] = float(lib["ay_raw"][chosen_idx])
            case["tau_raw"] = float(lib["tau_raw"][chosen_idx])

            for key in [
                "dv_kms",
                "ballistic_min_rM",
                "ballistic_corridor_dist",
                "ballistic_success",
                "row_index",
            ]:
                if key in lib and len(lib[key]) > chosen_idx:
                    case[key] = lib[key][chosen_idx]

        return case

    def _sample_ppo_b_case_from_library(self) -> Dict[str, Any]:
        """
        Sample one PPO-B library case according to label probabilities,
        or return one exact fixed index if configured.
        Supports both seed-based and state-based libraries.
        """
        self._load_ppo_b_scenario_library()
        lib = self.ppo_b_library_cache
        labels = np.asarray(lib["label"], dtype=np.int32)

        # --------------------------------------------------------
        # Fixed exact case mode
        # --------------------------------------------------------
        if bool(getattr(self.cfg, "ppo_b_use_fixed_index", False)):
            chosen_idx = int(getattr(self.cfg, "ppo_b_fixed_index", 0))
            return self._get_ppo_b_case_from_library_index(chosen_idx)

        idx_by_label: Dict[int, np.ndarray] = {
            0: np.where(labels == 0)[0],
            1: np.where(labels == 1)[0],
            2: np.where(labels == 2)[0],
        }

        probs = self._normalize_ppo_b_label_probs()

        available_labels = [k for k, idxs in idx_by_label.items() if len(idxs) > 0]
        if len(available_labels) == 0:
            raise ValueError("PPO-B scenario library contains no cases.")

        available_probs = np.array([probs[k] for k in available_labels], dtype=np.float64)
        if np.sum(available_probs) <= 0.0:
            available_probs = np.ones(len(available_labels), dtype=np.float64)
        available_probs /= np.sum(available_probs)

        chosen_label = int(self.rng.choice(np.array(available_labels, dtype=np.int32), p=available_probs))
        chosen_idx = int(self.rng.choice(idx_by_label[chosen_label]))

        fmt = str(
            np.asarray(
                lib.get("library_kind", lib.get("library_format", ["seed_commands"]))
            ).reshape(-1)[0]
        )

        case = {
            "index": chosen_idx,
            "label": chosen_label,
            "library_format": fmt,
        }

        if fmt == "handoff_states":
            case["state_handoff"] = np.asarray(lib["state_handoff"][chosen_idx], dtype=np.float64).copy()
            case["t_handoff"] = float(lib["t_handoff"][chosen_idx])

            for key in [
                "theta",
                "dv_kms",
                "term_reason",
                "success",
                "min_rM",
                "min_corridor_dist",
                "row_index",
                "state_after_tli",
                "t_after_tli",
                "rE_after_tli",
                "rM_after_tli",
                "dv0",
                "tli_theta",
                "spawn_theta",
            ]:
                if key in lib and len(lib[key]) > chosen_idx:
                    case[key] = lib[key][chosen_idx]
        else:
            case["theta"] = float(lib["theta"][chosen_idx])
            case["ax_raw"] = float(lib["ax_raw"][chosen_idx])
            case["ay_raw"] = float(lib["ay_raw"][chosen_idx])
            case["tau_raw"] = float(lib["tau_raw"][chosen_idx])

            for key in ["dv_kms", "ballistic_min_rM", "ballistic_corridor_dist", "ballistic_success"]:
                if key in lib and len(lib[key]) > chosen_idx:
                    v = lib[key][chosen_idx]
                    if np.isscalar(v):
                        case[key] = float(v) if key != "ballistic_success" else bool(v)

        return case

    def _apply_ppo_b_case_seed_noise(
        self,
        theta: float,
        ax_raw: float,
        ay_raw: float,
        tau_raw: float,
    ) -> Dict[str, float]:
        """
        Apply physical seed noise BEFORE building post-TLI state:
        - theta noise in degrees
        - TLI direction rotation noise in degrees
        - TLI magnitude noise in km/s
        """
        theta_noisy = float(theta)
        ax_noisy = float(ax_raw)
        ay_noisy = float(ay_raw)
        tau_noisy = float(tau_raw)

        # theta noise
        sigma_theta_deg = float(getattr(self.cfg, "ppo_b_noise_theta_deg", 0.0))
        if sigma_theta_deg > 0.0:
            theta_noisy += float(self.rng.normal(0.0, np.deg2rad(sigma_theta_deg)))
            theta_noisy = self._wrap_0_2pi(theta_noisy)

        # Build nominal raw DV vector in action space
        u = np.array([ax_noisy, ay_noisy], dtype=np.float64)
        u_norm = float(np.linalg.norm(u))
        if u_norm > 1.0:
            u = u / max(u_norm, 1e-12)

        # direction noise in degrees
        sigma_dir_deg = float(getattr(self.cfg, "ppo_b_noise_tli_dir_deg", 0.0))
        if sigma_dir_deg > 0.0 and np.linalg.norm(u) > 1e-12:
            dpsi = float(self.rng.normal(0.0, np.deg2rad(sigma_dir_deg)))
            u = self._rotate_vec_2d(u, dpsi)

        # magnitude noise in km/s, then convert relative to nondim TLI cap
        sigma_dv_kms = float(getattr(self.cfg, "ppo_b_noise_tli_dv_kms", 0.0))
        if sigma_dv_kms > 0.0 and np.linalg.norm(u) > 1e-12:
            dv_cap_nd = float(self._dv_cap_tli())
            u_mag = float(np.linalg.norm(u))
            dv_nom_nd = dv_cap_nd * u_mag
            dv_nom_kms = dv_nom_nd * cr3bp_vstar_kms()

            dv_noisy_kms = dv_nom_kms + float(self.rng.normal(0.0, sigma_dv_kms))
            dv_noisy_kms = max(0.0, dv_noisy_kms)
            dv_noisy_nd = dv_noisy_kms / max(cr3bp_vstar_kms(), 1e-12)

            # convert back to normalized raw action magnitude
            u_mag_new = dv_noisy_nd / max(dv_cap_nd, 1e-12)
            if u_mag_new > 1.0:
                u_mag_new = 1.0

            u_dir = u / max(np.linalg.norm(u), 1e-12)
            u = u_dir * u_mag_new

        ax_noisy = float(u[0])
        ay_noisy = float(u[1])

        return {
            "theta": theta_noisy,
            "ax_raw": ax_noisy,
            "ay_raw": ay_noisy,
            "tau_raw": tau_noisy,
        }

    def _build_post_tli_state_from_library_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a post-TLI payload from a sampled nominal TLI seed stored in the scenario library.
        This preserves the same payload shape expected by _set_post_tli_initial_condition().
        """
        seed = self._apply_ppo_b_case_seed_noise(
            theta=float(case["theta"]),
            ax_raw=float(case["ax_raw"]),
            ay_raw=float(case["ay_raw"]),
            tau_raw=float(case["tau_raw"]),
        )

        theta = float(seed["theta"])
        ax_raw = float(seed["ax_raw"])
        ay_raw = float(seed["ay_raw"])
        tau_raw = float(seed["tau_raw"])

        state0 = self._build_leo_state_from_theta(theta)

        old_state = self.state.copy()
        self.state = state0.copy()

        dv_cmd_nominal = self._dv_vec_from_action_xy(
            ax_raw,
            ay_raw,
            dv_cap=self._dv_cap_tli(),
        )
        dv_cmd_nominal = self._apply_pre_tli_control_mode(dv_cmd_nominal)
        dv_mag = float(np.linalg.norm(dv_cmd_nominal))

        state1 = state0.copy()
        state1[2] += dv_cmd_nominal[0]
        state1[3] += dv_cmd_nominal[1]

        dt_post = self._drift_time_from_tau_masked(tau_raw)
        state2 = self._propagate_copy(state1, dt_post)

        self.state = old_state

        payload = {
            "state_post_tli": state2,
            "state_immediate_after_burn": state1,
            "dv_vec_rot": dv_cmd_nominal.copy(),
            "dv_mag": float(dv_mag),
            "tau_raw": float(tau_raw),
            "ax_raw": float(ax_raw),
            "ay_raw": float(ay_raw),
            "theta": float(theta),
            "dt_post": float(dt_post),

            # metadata for logging/debug
            "scenario_index": int(case.get("index", -1)),
            "scenario_label": int(case.get("label", -1)),
            "scenario_ballistic_min_rM": float(case.get("ballistic_min_rM", np.nan)),
            "scenario_ballistic_corridor_dist": float(case.get("ballistic_corridor_dist", np.nan)),
            "scenario_ballistic_success": bool(case.get("ballistic_success", False)),
        }
        return payload

    # ------------------------------------------------------------
    # RESET + SPAWN CURRICULUM
    # ------------------------------------------------------------
    def _sample_spawn_theta(self, forced_theta: Optional[float] = None) -> float:
        if forced_theta is not None:
            return float(forced_theta)

        if not self.cfg.spawn_theta_limit_enabled:
            return float(self.rng.uniform(0.0, 2.0 * np.pi))

        a = float(self.cfg.spawn_theta_min)
        b = float(self.cfg.spawn_theta_max)

        if b < a:
            a, b = b, a

        return float(self.rng.uniform(a, b))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is None:
            options = {}

        if self.reward_model is not None and hasattr(self.reward_model, "reset_episode"):
            self.reward_model.reset_episode()

        mu = self.cfg.mu

        # --------------------------------------------------------
        # Base episode bookkeeping reset
        # --------------------------------------------------------
        self.t = 0.0
        self.state = np.zeros(4, dtype=np.float64)

        self.r_leo = float(self.cfg.r0_earth)
        self.r_leo_exit = float(self.cfg.left_leo_trigger_rE)
        self.t_orbit_leo_ref = self._leo_reference_orbit_period()
        self.pre_tli_timeout_limit = self._pre_tli_timeout_limit()

        self.steps = 0
        self.step_idx = 0

        self.ballistic_terminal_marker_rot = None
        self.terminal_marker_rot = None

        self.no_tli_terminated = False
        self.no_tli_termination_time = np.nan

        self.left_leo = False
        self.left_leo_step = None
        self.left_leo_time = np.nan

        self.flyby_done = False
        self.return_done = False
        self.success = False

        self._early_terminate = None
        self._substep_events = {
            "flyby": False,
            "corridor": False,
            "corridor_exit_outward": False,
        }

        self.min_rM = np.inf
        self.min_rE = np.inf
        self.min_rE_postflyby = np.inf

        self.tli_state_after_burn = None
        self.ballistic_ref_traj = []
        self.ballistic_ref_t_hist = []
        self.mcc_ballistic_overlays = []

        self.burn_events = []
        self.action_history = []
        self.t_hist = []
        
        
        

        self.return_corridor_hit_postflyby = False
        self.best_postflyby_corridor_dist = np.inf
        self.best_postflyby_rp = np.nan

        self.tli_used = False
        self.tli_executed = False
        self.tli_ballistic_reward_given = False
        self.mcc_burn_count = 0
        self.max_rE_seen_post_tli = 0.0
        self.invalid_return_armed_episode = False

        self.tli_tau = float("nan")
        self.tli_step_executed = -1
        self.tli_did_burn = False
        self.tli_ax = float("nan")
        self.tli_ay = float("nan")

        self.dv_used = 0.0
        self.dv0 = 0.0
        self.dv_mcc_total = 0.0

        self.pre_tli_cum_dv = 0.0
        self.pre_tli_burn_count = 0
        self.pre_tli_last_burn_mag = 0.0

        self.traj = []
        self.t_hist = []
        self.burns = []
        self.info_last = {}

        self.last_dt_effective = 0.0
        self.last_dt_warp = 0.0
        self.last_dt_post = 0.0
        self.last_tli_u01_raw = np.nan
        self.last_tli_u01_exec = np.nan

        self.ballistic_tli_reward_last = 0.0
        self.ballistic_tli_min_rM_last = np.nan
        self.ballistic_tli_min_rE_postflyby_last = np.nan
        self.ballistic_tli_corridor_dist_last = np.nan
        self.ballistic_tli_corridor_hit_last = False
        self.ballistic_tli_success_last = False
        self.ballistic_tli_vrel_at_min_rM_last = np.nan

        self.ppo_b_scenario_index = -1
        self.ppo_b_scenario_label = -1
        self.ppo_b_scenario_ballistic_min_rM = np.nan
        self.ppo_b_scenario_ballistic_corridor_dist = np.nan
        self.ppo_b_scenario_ballistic_success = False

        # --------------------------------------------------------
        # Spawn logic by trainer mode
        # --------------------------------------------------------
        trainer_mode = str(getattr(self.cfg, "trainer_mode", "ppo_a")).lower()

        if trainer_mode == "ppo_a":
            forced_spawn_theta = options.get("forced_spawn_theta", None)
            theta = self._sample_spawn_theta(forced_theta=forced_spawn_theta)
            self.state = self._build_leo_state_from_theta(theta)
            self.spawn_theta = float(theta)

        elif trainer_mode == "ppo_b_baseline":
            payload = self._build_post_tli_state_from_baseline()
            self._set_post_tli_initial_condition(payload)
            self.spawn_theta = float(payload.get("theta", np.nan))

        elif trainer_mode == "ppo_b_from_external_ic":
            payload = options.get("post_tli_init", None)
            if payload is None:
                raise ValueError(
                    "trainer_mode='ppo_b_from_external_ic' requires "
                    "reset(options={'post_tli_init': payload})"
                )
            self._set_post_tli_initial_condition(payload)
            self.spawn_theta = float(payload.get("theta", np.nan))

        elif trainer_mode == "ppo_b_library":
            case = self._sample_ppo_b_case_from_library()

            if str(case.get("library_format", "seed")) == "handoff_states":
                payload = self._build_post_tli_state_from_handoff_case(case)
            else:
                payload = self._build_post_tli_state_from_library_case(case)

            self._set_post_tli_initial_condition(payload)

            self.ppo_b_scenario_index = int(payload.get("scenario_index", -1))
            self.ppo_b_scenario_label = int(payload.get("scenario_label", -1))
            self.ppo_b_scenario_ballistic_min_rM = float(payload.get("scenario_ballistic_min_rM", np.nan))
            self.ppo_b_scenario_ballistic_corridor_dist = float(payload.get("scenario_ballistic_corridor_dist", np.nan))
            self.ppo_b_scenario_ballistic_success = bool(payload.get("scenario_ballistic_success", False))

            self.spawn_theta = float(payload.get("theta", np.nan))

        # --------------------------------------------------------
        # Final per-reset geometry/log init
        # --------------------------------------------------------
        rE0, rM0 = dist_to_primaries(mu, self.state)
        self.prev_rE = float(rE0)
        self.prev_rM = float(rM0)

        if rE0 >= self.r_leo_exit:
            self.left_leo = True
            self.left_leo_step = 0
            self.left_leo_time = float(self.t)

        self.traj = [self.state.copy()]
        self.t_hist = [float(self.t)]

        obs = self._get_obs()
        info = self._get_info(extra={"term_reason": "reset", "left_leo": bool(self.left_leo)})
        return obs, info

    # ------------------------------------------------------------
    # PASTE THESE UNCHANGED FROM V2_9
    # ------------------------------------------------------------
    # - _propagate
    # - _propagate_copy
    # - _consume_substep_events
    # - _build_ballistic_reference_from_tli
    # - _dv_cap_tli
    # - _dv_cap_mcc
    # - _tau_from_action
    # - _mcc_dv_from_u
    # - _update_mcc_gate_availability
    # - _get_active_mcc_slot
    # - _timewarp_to_tau_angle

    def _propagate(self, dt_total: float):
        dt_total = float(dt_total)
        if dt_total <= 0.0:
            return

        mu = self.cfg.mu
        rE_pos, rM_pos = earth_moon_positions(mu)

        t_remaining = dt_total

        while t_remaining > 0.0:
            # Region-aware RK4 target substep:
            # use fine stepping near Earth/Moon, otherwise use normal adaptive target
            if in_fine_integration_region(self.cfg, self.state):
                dt_sub_target = fine_rk4_substep_nondim()
            else:
                dt_sub_target = rk4_target_substep_nondim(t_remaining)

            dt_sub = min(t_remaining, max(dt_sub_target, 1e-12))

            self.state = rk4_step(self.cfg.mu, self.state, dt_sub)
            self.t += dt_sub
            t_remaining -= dt_sub

            pos = self.state[:2]
            rE_now = float(np.linalg.norm(pos - rE_pos))
            rM_now = float(np.linalg.norm(pos - rM_pos))

            self.min_rE = min(self.min_rE, rE_now)
            self.min_rM = min(self.min_rM, rM_now)

            if self.flyby_done:
                self.min_rE_postflyby = min(self.min_rE_postflyby, rE_now)

                corridor_dist_now = dist_to_interval(
                    float(rE_now),
                    float(self.cfg.rp_min),
                    float(self.cfg.rp_max),
                )
                self.best_postflyby_corridor_dist = min(
                    float(self.best_postflyby_corridor_dist),
                    float(corridor_dist_now),
                )

                if corridor_dist_now <= 0.0:
                    self.best_postflyby_rp = float(rE_now)

            if (not self.flyby_done) and (rM_now <= self.cfg.r_moon_flyby):
                self._substep_events["flyby"] = True

            if self.flyby_done or self._substep_events["flyby"]:
                inside_corridor = bool(self.cfg.rp_min <= rE_now <= self.cfg.rp_max)

                if inside_corridor:
                    self._substep_events["corridor"] = True

                corridor_seen = bool(
                    self.return_corridor_hit_postflyby
                    or self._substep_events["corridor"]
                )

                # Success candidate becomes real success only after outward exit.
                # Outward exit means rE > rp_max.
                # Inward exit toward Earth must not count as success.
                if corridor_seen and (rE_now > self.cfg.rp_max):
                    self._substep_events["corridor_exit_outward"] = True

            if rE_now <= self.cfg.r_earth_impact:
                self._early_terminate = ("earth_impact", rE_now)
                return

            if rM_now <= self.cfg.r_moon_impact:
                self._early_terminate = ("moon_impact", rM_now)
                return

            if self.debug_eval or self.cfg.store_dense_training_traj:
                self.traj.append(self.state.copy())
                self.t_hist.append(float(self.t))

    def _propagate_copy(self, state: np.ndarray, dt_total: float) -> np.ndarray:
        dt_total = float(dt_total)
        s = np.array(state, dtype=np.float64).copy()

        if dt_total <= 0.0:
            return s

        t_remaining = dt_total

        while t_remaining > 0.0:
            if in_fine_integration_region(self.cfg, s):
                dt_sub_target = fine_rk4_substep_nondim()
            else:
                dt_sub_target = rk4_target_substep_nondim(t_remaining)

            dt_sub = min(t_remaining, max(dt_sub_target, 1e-12))
            s = rk4_step(self.cfg.mu, s, dt_sub)
            t_remaining -= dt_sub

        return s

    def _consume_substep_events(self):
        if self._substep_events["flyby"]:
            self.flyby_done = True

        if self._substep_events["corridor"]:
            self.return_corridor_hit_postflyby = True
            self.return_done = True

        if self._substep_events["corridor_exit_outward"]:
            self.return_done = True
            self.success = True

        self._substep_events["flyby"] = False
        self._substep_events["corridor"] = False
        self._substep_events["corridor_exit_outward"] = False

    def _build_ballistic_reference_from_tli(self):
        self.ballistic_ref_traj = []
        self.ballistic_ref_t_hist = []

        if self.tli_state_after_burn is None:
            return

        s = np.array(self.tli_state_after_burn, dtype=np.float64).copy()
        t_ref = float(self.t)

        rE_pos, rM_pos = earth_moon_positions(self.cfg.mu)

        self.ballistic_ref_traj.append(s.copy())
        self.ballistic_ref_t_hist.append(t_ref)

        dt = float(self.cfg.dt)
        t_end = float(self.cfg.t_max)

        impacted = False

        while t_ref < t_end and not impacted:
            n_sub = int(getattr(self.cfg, "integration_substeps", 1))
            n_sub = max(1, n_sub)
            dt_sub = dt / n_sub

            for _ in range(n_sub):
                s = rk4_step(self.cfg.mu, s, dt_sub)
                t_ref += dt_sub

                self.ballistic_ref_traj.append(s.copy())
                self.ballistic_ref_t_hist.append(t_ref)

                pos = s[:2].astype(np.float64)
                rE = float(np.linalg.norm(pos - rE_pos))
                rM = float(np.linalg.norm(pos - rM_pos))

                if rE <= self.cfg.r_earth_impact:
                    self.ballistic_terminal_marker_rot = pos.copy()
                    impacted = True
                    break

                if rM <= self.cfg.r_moon_impact:
                    self.ballistic_terminal_marker_rot = pos.copy()
                    impacted = True
                    break

                if t_ref >= t_end:
                    break

    def _dv_cap_tli(self) -> float:
        if RUN.use_global_burn_cap_kms:
            return float(global_burn_cap_nondim())
        if RUN.use_single_dv_cap:
            return float(RUN.dv_cap_single)
        return float(self.cfg.dv_max_tli)

    def _dv_cap_mcc(self) -> float:
        if RUN.use_global_burn_cap_kms:
            return float(global_burn_cap_nondim())
        if RUN.use_single_dv_cap:
            return float(RUN.dv_cap_single)
        return float(self.cfg.dv_max_mcc)


    def _pre_tli_mode(self) -> bool:
        return bool(not self.tli_used)

    def _current_tau_bounds_minutes(self) -> Tuple[float, float]:
        if self._pre_tli_mode():
            return (
                float(RUN.drift_min_minutes_pre_tli),
                float(RUN.drift_max_minutes_pre_tli),
            )
        return (
            float(RUN.drift_min_minutes_post_tli),
            float(RUN.drift_max_minutes_post_tli),
        )

    def _current_tau_bounds_nondim(self) -> Tuple[float, float]:
        tau_min_min, tau_max_min = self._current_tau_bounds_minutes()
        dt_min = minutes_to_nondim_time(tau_min_min)
        dt_max = minutes_to_nondim_time(tau_max_min)
        if dt_max < dt_min:
            dt_min, dt_max = dt_max, dt_min
        return float(dt_min), float(dt_max)

    def _drift_time_from_tau_masked(self, tau_raw: float) -> float:
        tau01 = 0.5 * (float(tau_raw) + 1.0)
        tau01 = float(np.clip(tau01, 0.0, 1.0))

        dt_min, dt_max = self._current_tau_bounds_nondim()
        return float(dt_min + tau01 * (dt_max - dt_min))

    def _current_dv_cap(self) -> float:
        if self._pre_tli_mode():
            return float(self._dv_cap_tli())
        return float(self._dv_cap_mcc())

    def _pre_tli_burn_deadzone(self) -> float:
        return float(RUN.pre_tli_burn_deadzone_frac_of_tli_cap) * float(self._dv_cap_tli())


    def _leo_reference_orbit_period(self) -> float:
        muE = 1.0 - float(self.cfg.mu)
        r0 = float(self.cfg.r0_earth)
        return float(2.0 * np.pi * np.sqrt((r0 ** 3) / max(muE, 1e-12)))

    def _pre_tli_timeout_limit(self) -> float:
        return float(RUN.no_tli_terminate_after_leo_orbits) * float(self.t_orbit_leo_ref)
    
    def _left_leo_no_tli_grace_nondim(self) -> float:
        return float(minutes_to_nondim_time(self.cfg.left_leo_no_tli_grace_minutes))

    

    def _dv_vec_from_action_xy(self, ax_raw: float, ay_raw: float, dv_cap: float) -> np.ndarray:
        """
        Stephenson-like direct burn-vector mapping in 2D.

        The policy outputs a raw planar burn vector [ax, ay] in [-1,1]^2.
        We interpret that vector directly as the commanded DV direction/magnitude,
        then project it onto the unit disk so that the applied burn satisfies

            ||dv_cmd|| <= dv_cap

        This is the planar analogue of an action in Δv_max * B^3.
        """
        u = np.array([float(ax_raw), float(ay_raw)], dtype=np.float64)
        u_norm = float(np.linalg.norm(u))

        if u_norm > 1.0:
            u = u / u_norm

        return float(dv_cap) * u



    # ------------------------------------------------------------
    # V3: EXECUTION NOISE
    # ------------------------------------------------------------
    def _apply_burn_execution_noise(self, dv_cmd_nominal: np.ndarray, burn_kind: str) -> np.ndarray:
        dv_cmd_nominal = np.asarray(dv_cmd_nominal, dtype=np.float64).reshape(2,)

        if burn_kind == "TLI":
            sigma = float(self.cfg.dv_noise_sigma_tli)
        else:
            sigma = float(self.cfg.dv_noise_sigma_mcc)

        if sigma <= 0.0:
            return dv_cmd_nominal.copy()

        noise = self.rng.normal(loc=0.0, scale=sigma, size=2).astype(np.float64)
        return dv_cmd_nominal + noise

    # ------------------------------------------------------------
    # V3: BALLISTIC TLI EVALUATION
    # ------------------------------------------------------------
    def _evaluate_ballistic_after_tli(self, state_after_tli: np.ndarray, t_after_tli: float) -> Dict[str, Any]:
        s = np.array(state_after_tli, dtype=np.float64).copy()
        t_local = float(t_after_tli)
        self.ballistic_terminal_marker_rot = None

        mu = float(self.cfg.mu)
        rE_pos, rM_pos = earth_moon_positions(mu)

        min_rM = np.inf
        vrel_at_min_rM = np.nan
        flyby_done_local = False

        min_rE_postflyby = np.inf
        corridor_hit_postflyby = False
        corridor_exit_outward = False

        # Track whether the ballistic trajectory actually leaves LEO
        rE0 = float(np.linalg.norm(s[:2] - rE_pos))
        left_leo_local = bool(self.left_leo or (rE0 > getattr(self, "r_leo_exit", np.inf)))

        # Track farthest Earth distance reached during ballistic propagation
        max_rE_seen = float(rE0)

        # Invalid pre-flyby Earth-return detector state
        invalid_return_armed = False

        dt = float(self.cfg.dt)

        term_reason = "timeout"
        success = False
        rE_terminal = np.nan
        rM_terminal = np.nan

        # Budget exceed check is trivial here because ballistic rollout adds no more DV.
        dv_used_ballistic = float(self.dv_used)

        while t_local < float(self.cfg.t_max):
            n_sub = max(1, int(self.cfg.integration_substeps))
            dt_sub = dt / n_sub

            for _ in range(n_sub):
                s = rk4_step(self.cfg.mu, s, dt_sub)
                t_local += dt_sub

                pos = s[:2].astype(np.float64)
                vel = s[2:4].astype(np.float64)

                rE = float(np.linalg.norm(pos - rE_pos))
                rM = float(np.linalg.norm(pos - rM_pos))
                rb = float(np.linalg.norm(pos))
                max_rE_seen = max(max_rE_seen, rE)

                # Earth-centered inertial radial velocity
                v_sc_I = v_rot_to_inertial(pos, vel)
                v_earth_I = omega_cross_r_2d(rE_pos)
                r_sc_E = pos - rE_pos
                v_sc_E_I = v_sc_I - v_earth_I
                vrE = radial_velocity_about_point(r_sc_E, v_sc_E_I)

                if rE > getattr(self, "r_leo_exit", np.inf):
                    left_leo_local = True

                # Arm invalid-return detection only after a meaningful outbound branch
                if (
                    bool(self.cfg.ballistic_invalid_preflyby_return_enabled)
                    and (not flyby_done_local)
                    and (rE >= float(self.cfg.ballistic_invalid_return_arm_rE))
                ):
                    invalid_return_armed = True

                if rM < min_rM:
                    min_rM = rM

                    v_sc_I = v_rot_to_inertial(pos, vel)
                    v_moon_I = omega_cross_r_2d(rM_pos)
                    v_rel_M = v_sc_I - v_moon_I
                    vrel_at_min_rM = float(np.linalg.norm(v_rel_M))

                if (not flyby_done_local) and (rM <= self.cfg.r_moon_flyby):
                    flyby_done_local = True

                if flyby_done_local:
                    min_rE_postflyby = min(min_rE_postflyby, rE)

                    if self.cfg.rp_min <= rE <= self.cfg.rp_max:
                        corridor_hit_postflyby = True

                    if corridor_hit_postflyby and (rE > self.cfg.rp_max):
                        corridor_exit_outward = True
                
                

                # Invalid ballistic branch, case 2:
                # got meaningfully outbound, but is now clearly falling back to Earth
                # while still far from the Moon
                if (
                    bool(self.cfg.ballistic_invalid_preflyby_return_enabled)
                    and invalid_return_armed
                    and (not flyby_done_local)
                    and (vrE <= float(self.cfg.ballistic_invalid_return_vrE_threshold))
                    and (rM > float(self.cfg.ballistic_invalid_return_moon_far_rM))
                ):
                    term_reason = "invalid_preflyby_earth_return"
                    rE_terminal = rE
                    rM_terminal = rM

                    self.ballistic_terminal_marker_rot = pos.copy()

                    return {
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": float(dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max))
                        if np.isfinite(min_rE_postflyby) else np.nan,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                # Match the same terminal logic categories as the real env
                if rE <= self.cfg.r_earth_impact:
                    term_reason = "earth_impact"
                    rE_terminal = rE
                    rM_terminal = rM
                    self.ballistic_terminal_marker_rot = pos.copy()
                    return {
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": float(dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max))
                        if np.isfinite(min_rE_postflyby) else np.nan,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                if rM <= self.cfg.r_moon_impact:
                    term_reason = "moon_impact"
                    rE_terminal = rE
                    rM_terminal = rM
                    self.ballistic_terminal_marker_rot = pos.copy()
                    return {
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": float(dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max))
                        if np.isfinite(min_rE_postflyby) else np.nan,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                if rb >= self.cfg.r_escape:
                    term_reason = "escape"
                    rE_terminal = rE
                    rM_terminal = rM
                    self.ballistic_terminal_marker_rot = pos.copy()
                    return {
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": float(dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max))
                        if np.isfinite(min_rE_postflyby) else np.nan,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                if self.cfg.terminate_on_dv_budget_exceed and (self.reward_model is not None):
                    dv_budget = float(self.reward_model.cfg.dv_budget)
                    if dv_used_ballistic > dv_budget:
                        term_reason = "dv_budget_exceeded"
                        rE_terminal = rE
                        rM_terminal = rM
                        self.ballistic_terminal_marker_rot = pos.copy()
                        return {
                            "min_rM": float(min_rM),
                            "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                            "flyby_done": bool(flyby_done_local),
                            "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                            "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                            "corridor_dist": float(dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max))
                            if np.isfinite(min_rE_postflyby) else np.nan,
                            "term_reason": term_reason,
                            "success": False,
                            "rE_terminal": float(rE_terminal),
                            "rM_terminal": float(rM_terminal),
                            "left_leo": bool(left_leo_local),
                            "dv_used_ballistic": float(dv_used_ballistic),
                        }

                if t_local >= float(self.cfg.t_max):
                    break

            if t_local >= float(self.cfg.t_max):
                break

        # If ballistic rollout survives to t_max, classify final outcome
        if (
            bool(self.cfg.ballistic_invalid_preflyby_return_enabled)
            and (not flyby_done_local)
            and (max_rE_seen < float(self.cfg.ballistic_invalid_min_meaningful_outbound_rE))
        ):
            term_reason = "invalid_preflyby_earth_return"
            success = False
        elif corridor_exit_outward:
                term_reason = "success"
                success = True
        else:
            term_reason = "timeout"
            success = False

        self.ballistic_terminal_marker_rot = s[:2].copy()

        rE_final = float(np.linalg.norm(s[:2] - rE_pos))
        rM_final = float(np.linalg.norm(s[:2] - rM_pos))

        corridor_dist = np.inf
        if np.isfinite(min_rE_postflyby):
            corridor_dist = dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max)
        return {
            "min_rM": float(min_rM),
            "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
            "flyby_done": bool(flyby_done_local),
            "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
            "corridor_hit_postflyby": bool(corridor_hit_postflyby),
            "corridor_dist": float(corridor_dist) if np.isfinite(corridor_dist) else np.nan,
            "term_reason": term_reason,
            "success": bool(success),
            "rE_terminal": float(rE_final),
            "rM_terminal": float(rM_final),
            "left_leo": bool(left_leo_local),
            "dv_used_ballistic": float(dv_used_ballistic),
        }


    def _evaluate_ballistic_overlay_from_state(
        self,
        state_start: np.ndarray,
        t_start: float,
    ) -> Dict[str, Any]:
        """
        Ballistically propagate from an arbitrary post-burn state and
        record the full branch for later plotting.

        This is used for MCC diagnostics:
        "If we stop controlling right after this MCC, what happens?"
        """
        s = np.array(state_start, dtype=np.float64).copy()
        t_local = float(t_start)

        mu = float(self.cfg.mu)
        rE_pos, rM_pos = earth_moon_positions(mu)

        traj = [s.copy()]
        t_hist = [float(t_local)]

        min_rM = np.inf
        vrel_at_min_rM = np.nan

        flyby_done_local = bool(self.flyby_done)
        min_rE_postflyby = float(self.min_rE_postflyby) if bool(self.flyby_done) else np.inf
        corridor_hit_postflyby = bool(self.return_corridor_hit_postflyby)
        corridor_exit_outward = False

        rE0 = float(np.linalg.norm(s[:2] - rE_pos))
        left_leo_local = bool(self.left_leo or (rE0 > getattr(self, "r_leo_exit", np.inf)))
        max_rE_seen = float(rE0)

        invalid_return_armed = bool(self.invalid_return_armed_episode)

        dt = float(self.cfg.dt)

        term_reason = "timeout"
        success = False
        rE_terminal = np.nan
        rM_terminal = np.nan
        terminal_marker_rot = None

        dv_used_ballistic = float(self.dv_used)

        while t_local < float(self.cfg.t_max):
            n_sub = max(1, int(self.cfg.integration_substeps))
            dt_sub = dt / n_sub

            for _ in range(n_sub):
                s = rk4_step(self.cfg.mu, s, dt_sub)
                t_local += dt_sub

                traj.append(s.copy())
                t_hist.append(float(t_local))

                pos = s[:2].astype(np.float64)
                vel = s[2:4].astype(np.float64)

                rE = float(np.linalg.norm(pos - rE_pos))
                rM = float(np.linalg.norm(pos - rM_pos))
                rb = float(np.linalg.norm(pos))
                max_rE_seen = max(max_rE_seen, rE)

                v_sc_I = v_rot_to_inertial(pos, vel)
                v_earth_I = omega_cross_r_2d(rE_pos)
                r_sc_E = pos - rE_pos
                v_sc_E_I = v_sc_I - v_earth_I
                vrE = radial_velocity_about_point(r_sc_E, v_sc_E_I)

                if rE > getattr(self, "r_leo_exit", np.inf):
                    left_leo_local = True

                if (
                    bool(self.cfg.ballistic_invalid_preflyby_return_enabled)
                    and (not flyby_done_local)
                    and (rE >= float(self.cfg.ballistic_invalid_return_arm_rE))
                ):
                    invalid_return_armed = True

                if rM < min_rM:
                    min_rM = rM

                    v_sc_I = v_rot_to_inertial(pos, vel)
                    v_moon_I = omega_cross_r_2d(rM_pos)
                    v_rel_M = v_sc_I - v_moon_I
                    vrel_at_min_rM = float(np.linalg.norm(v_rel_M))

                if (not flyby_done_local) and (rM <= self.cfg.r_moon_flyby):
                    flyby_done_local = True

                if flyby_done_local:
                    min_rE_postflyby = min(min_rE_postflyby, rE)

                    if self.cfg.rp_min <= rE <= self.cfg.rp_max:
                        corridor_hit_postflyby = True

                    if corridor_hit_postflyby and (rE > self.cfg.rp_max):
                        corridor_exit_outward = True

                if (
                    bool(self.cfg.ballistic_invalid_preflyby_return_enabled)
                    and invalid_return_armed
                    and (not flyby_done_local)
                    and (vrE <= float(self.cfg.ballistic_invalid_return_vrE_threshold))
                    and (rM > float(self.cfg.ballistic_invalid_return_moon_far_rM))
                ):
                    term_reason = "invalid_preflyby_earth_return"
                    rE_terminal = rE
                    rM_terminal = rM
                    terminal_marker_rot = pos.copy()

                    corridor_dist = np.nan
                    if np.isfinite(min_rE_postflyby):
                        corridor_dist = float(
                            dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max)
                        )

                    return {
                        "traj_rot": np.asarray(traj, dtype=np.float64),
                        "t_hist": np.asarray(t_hist, dtype=np.float64),
                        "terminal_marker_rot": terminal_marker_rot,
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": corridor_dist,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                if rE <= self.cfg.r_earth_impact:
                    term_reason = "earth_impact"
                    rE_terminal = rE
                    rM_terminal = rM
                    terminal_marker_rot = pos.copy()

                    corridor_dist = np.nan
                    if np.isfinite(min_rE_postflyby):
                        corridor_dist = float(
                            dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max)
                        )

                    return {
                        "traj_rot": np.asarray(traj, dtype=np.float64),
                        "t_hist": np.asarray(t_hist, dtype=np.float64),
                        "terminal_marker_rot": terminal_marker_rot,
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": corridor_dist,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                if rM <= self.cfg.r_moon_impact:
                    term_reason = "moon_impact"
                    rE_terminal = rE
                    rM_terminal = rM
                    terminal_marker_rot = pos.copy()

                    corridor_dist = np.nan
                    if np.isfinite(min_rE_postflyby):
                        corridor_dist = float(
                            dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max)
                        )

                    return {
                        "traj_rot": np.asarray(traj, dtype=np.float64),
                        "t_hist": np.asarray(t_hist, dtype=np.float64),
                        "terminal_marker_rot": terminal_marker_rot,
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": corridor_dist,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                if rb >= self.cfg.r_escape:
                    term_reason = "escape"
                    rE_terminal = rE
                    rM_terminal = rM
                    terminal_marker_rot = pos.copy()

                    corridor_dist = np.nan
                    if np.isfinite(min_rE_postflyby):
                        corridor_dist = float(
                            dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max)
                        )

                    return {
                        "traj_rot": np.asarray(traj, dtype=np.float64),
                        "t_hist": np.asarray(t_hist, dtype=np.float64),
                        "terminal_marker_rot": terminal_marker_rot,
                        "min_rM": float(min_rM),
                        "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                        "flyby_done": bool(flyby_done_local),
                        "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                        "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                        "corridor_dist": corridor_dist,
                        "term_reason": term_reason,
                        "success": False,
                        "rE_terminal": float(rE_terminal),
                        "rM_terminal": float(rM_terminal),
                        "left_leo": bool(left_leo_local),
                        "dv_used_ballistic": float(dv_used_ballistic),
                    }

                if self.cfg.terminate_on_dv_budget_exceed and (self.reward_model is not None):
                    dv_budget = float(self.reward_model.cfg.dv_budget)
                    if dv_used_ballistic > dv_budget:
                        term_reason = "dv_budget_exceeded"
                        rE_terminal = rE
                        rM_terminal = rM
                        terminal_marker_rot = pos.copy()

                        corridor_dist = np.nan
                        if np.isfinite(min_rE_postflyby):
                            corridor_dist = float(
                                dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max)
                            )

                        return {
                            "traj_rot": np.asarray(traj, dtype=np.float64),
                            "t_hist": np.asarray(t_hist, dtype=np.float64),
                            "terminal_marker_rot": terminal_marker_rot,
                            "min_rM": float(min_rM),
                            "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
                            "flyby_done": bool(flyby_done_local),
                            "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
                            "corridor_hit_postflyby": bool(corridor_hit_postflyby),
                            "corridor_dist": corridor_dist,
                            "term_reason": term_reason,
                            "success": False,
                            "rE_terminal": float(rE_terminal),
                            "rM_terminal": float(rM_terminal),
                            "left_leo": bool(left_leo_local),
                            "dv_used_ballistic": float(dv_used_ballistic),
                        }

                if t_local >= float(self.cfg.t_max):
                    break

            if t_local >= float(self.cfg.t_max):
                break

        if (
            bool(self.cfg.ballistic_invalid_preflyby_return_enabled)
            and (not flyby_done_local)
            and (max_rE_seen < float(self.cfg.ballistic_invalid_min_meaningful_outbound_rE))
        ):
            term_reason = "invalid_preflyby_earth_return"
            success = False
        elif corridor_exit_outward:
            term_reason = "success"
            success = True
        else:
            term_reason = "timeout"
            success = False

        terminal_marker_rot = s[:2].copy()

        rE_final = float(np.linalg.norm(s[:2] - rE_pos))
        rM_final = float(np.linalg.norm(s[:2] - rM_pos))

        corridor_dist = np.nan
        if np.isfinite(min_rE_postflyby):
            corridor_dist = float(
                dist_to_interval(float(min_rE_postflyby), self.cfg.rp_min, self.cfg.rp_max)
            )

        return {
            "traj_rot": np.asarray(traj, dtype=np.float64),
            "t_hist": np.asarray(t_hist, dtype=np.float64),
            "terminal_marker_rot": terminal_marker_rot,
            "min_rM": float(min_rM),
            "vrel_at_min_rM": float(vrel_at_min_rM) if np.isfinite(vrel_at_min_rM) else np.nan,
            "flyby_done": bool(flyby_done_local),
            "min_rE_postflyby": float(min_rE_postflyby) if np.isfinite(min_rE_postflyby) else np.nan,
            "corridor_hit_postflyby": bool(corridor_hit_postflyby),
            "corridor_dist": corridor_dist,
            "term_reason": term_reason,
            "success": bool(success),
            "rE_terminal": float(rE_final),
            "rM_terminal": float(rM_final),
            "left_leo": bool(left_leo_local),
            "dv_used_ballistic": float(dv_used_ballistic),
        }

    def _maybe_store_mcc_ballistic_overlay(
        self,
        state_after_burn: np.ndarray,
        t_after_burn: float,
        burn_event: Dict[str, Any],
    ) -> None:
        """
        Build and store one MCC ballistic overlay branch for later plotting.

        Guarded so this only runs in eval/debug episodes when enabled.
        """
        if not bool(getattr(RUN, "generate_mcc_eval_plot", False)):
            return

        if not bool(self.debug_eval):
            return

        dv_mag = float(burn_event.get("dv_mag", 0.0))
        dv_min_nd = float(kms_to_nondim_dv(getattr(RUN, "mcc_overlay_min_dv_kms", 0.0)))

        if dv_mag <= max(0.0, dv_min_nd):
            return

        metrics = self._evaluate_ballistic_overlay_from_state(
            state_start=np.asarray(state_after_burn, dtype=np.float64),
            t_start=float(t_after_burn),
        )

        overlay = {
            "burn_index": int(len(self.mcc_ballistic_overlays) + 1),
            "burn_kind": str(burn_event.get("kind", "POST_TLI")),
            "burn_time": float(burn_event.get("time", t_after_burn)),
            "burn_step_idx": int(burn_event.get("step_idx", -1)),
            "dv_mag": float(dv_mag),
            "traj_rot": np.asarray(metrics["traj_rot"], dtype=np.float64),
            "t_hist": np.asarray(metrics["t_hist"], dtype=np.float64),
            "terminal_marker_rot": None
            if metrics.get("terminal_marker_rot", None) is None
            else np.asarray(metrics["terminal_marker_rot"], dtype=np.float64).copy(),
            "term_reason": str(metrics.get("term_reason", "")),
            "success": bool(metrics.get("success", False)),
            "corridor_dist": float(metrics.get("corridor_dist", np.nan)),
            "min_rM": float(metrics.get("min_rM", np.nan)),
            "flyby_done": bool(metrics.get("flyby_done", False)),
            "corridor_hit_postflyby": bool(metrics.get("corridor_hit_postflyby", False)),
        }

        self.mcc_ballistic_overlays.append(overlay)

    def _compute_ballistic_tli_reward(self, metrics: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate ballistic-after-TLI reward using a ballistic-only proxy rollout.

        This helper reward is based only on what the coast-only ballistic trajectory
        would do after the committed TLI:
        - flyby reward
        - return reward
        - escape penalty
        - crash penalty
        - optional velocity-at-flyby term

        Budget penalty is intentionally excluded from the ballistic helper.
        If the real episode already exceeds budget, the ballistic helper is skipped
        and the real env handles the full budget penalty.
        """
        terms: Dict[str, float] = {}

        if self.reward_model is None:
            return 0.0, terms

        # Build a temporary reward model so we can reuse the real reward logic
        temp_rm = SeanStyleReward(self.reward_model.cfg, self.reward_model.w)
        temp_rm.reset_episode()

        # Seed the temporary reward model with ballistic closest-approach data
        min_rM = float(metrics.get("min_rM", np.inf))
        vrel_at_min = float(metrics.get("vrel_at_min_rM", np.nan))

        if np.isfinite(min_rM):
            temp_rm.min_rM = float(min_rM)
        if np.isfinite(vrel_at_min):
            temp_rm.v_at_min_rM = float(vrel_at_min)

        term_reason = str(metrics.get("term_reason", "timeout"))
        terminated = term_reason in (
            "earth_impact",
            "moon_impact",
            "escape",
            "dv_budget_exceeded",
            "invalid_preflyby_earth_return",
        )
        truncated = term_reason in ("timeout", "success")

        ballistic_info = {
            "term_reason": term_reason,
            "rE": float(metrics.get("rE_terminal", np.inf)),
            "rM": float(metrics.get("rM_terminal", np.inf)),
            "vrel_moon": float(metrics.get("vrel_at_min_rM", 0.0)),
            "dv_used": float(metrics.get("dv_used_ballistic", self.dv_used)),
            "min_rM": float(metrics.get("min_rM", np.inf)),
            "vrel_at_min_rM": float(metrics.get("vrel_at_min_rM", np.nan)),
            "min_rE_postflyby": float(metrics.get("min_rE_postflyby", np.inf)),
            "flyby_done": bool(metrics.get("flyby_done", False)),
            "return_corridor_hit_postflyby": bool(metrics.get("corridor_hit_postflyby", False)),
            "left_leo": bool(metrics.get("left_leo", self.left_leo)),
            "success": bool(metrics.get("success", False)),
        }

        raw_reward, raw_terms = temp_rm.compute(
            env=self,
            state=self.state,
            info=ballistic_info,
            dv_mag=0.0,
            terminated=terminated,
            truncated=truncated,
        )

        # Repackage under ballistic-specific names so your logging stays clear
        terms["r_tli_ballistic_dv"] = 0.0
        terms["r_tli_ballistic_budget"] = 0.0
        terms["r_tli_ballistic_escape"] = float(raw_terms.get("r_escape", 0.0))
        terms["r_tli_ballistic_invalid_preflyby_earth_return"] = float(
            raw_terms.get("r_invalid_preflyby_earth_return", 0.0)
        )
        terms["r_tli_ballistic_crash"] = float(raw_terms.get("r_crash", 0.0))
        terms["r_tli_ballistic_flyby"] = float(raw_terms.get("r_flyby", 0.0))
        terms["r_tli_ballistic_velocity"] = float(raw_terms.get("r_velocity", 0.0))
        terms["r_tli_ballistic_return"] = float(raw_terms.get("r_return", 0.0))


        # Helpful diagnostics
        terms["r_tli_ballistic_terminal_rewards_suppressed"] = float(raw_terms.get("terminal_rewards_suppressed", 0.0))
        terms["r_tli_ballistic_terminal_escape_flag"] = float(raw_terms.get("terminal_escape", 0.0))
        terms["r_tli_ballistic_terminal_timeout_flag"] = float(raw_terms.get("terminal_timeout", 0.0))
        terms["r_tli_ballistic_terminal_success_flag"] = float(raw_terms.get("terminal_success", 0.0))
        terms["r_tli_ballistic_term_reason_code"] = 0.0  # diagnostic placeholder

        # Apply the same ballistic scale to EVERYTHING in this proxy reward
        scale = float(np.clip(RUN.tli_ballistic_scale, 0.0, 1.0))

        # Remove any ballistic budget penalty from the helper reward.
        allowed_raw_total = (
            float(raw_terms.get("r_escape", 0.0))
            + float(raw_terms.get("r_invalid_preflyby_earth_return", 0.0))
            + float(raw_terms.get("r_crash", 0.0))
            + float(raw_terms.get("r_flyby", 0.0))
            + float(raw_terms.get("r_velocity", 0.0))
            + float(raw_terms.get("r_return", 0.0))
        )

        for k in list(terms.keys()):
            terms[k] = float(terms[k]) * scale

        total = float(allowed_raw_total) * scale
        terms["r_tli_ballistic_total"] = float(total)
        terms["r_tli_ballistic_scale"] = float(scale)

        return float(total), terms

    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        cfg = self.cfg
        mu = cfg.mu
        obs_before_action = self._get_obs().copy()
        state_before_action = np.asarray(self.state, dtype=np.float64).copy()
        t_before_action = float(self.t)

        self._early_terminate = None
        self._substep_events["flyby"] = False
        self._substep_events["corridor"] = False
        self._substep_events["corridor_exit_outward"] = False

        self.last_dt_effective = 0.0
        self.last_dt_warp = 0.0
        self.last_dt_post = 0.0

        if self._use_tangential_scalar_action():
            if action.size != 2:
                raise ValueError(f"Expected 2D action [dv_tan_raw, tau_raw], got shape {action.shape}")
            dv_tan_raw = float(action[0])
            tau_raw = float(action[1])

            ax_raw = float(dv_tan_raw)   # keep logging field populated
            ay_raw = 0.0                 # dummy for logging compatibility

            raw_dv_dirmag = np.array([dv_tan_raw, 0.0], dtype=np.float64)
            raw_dv_dirmag_norm = abs(float(dv_tan_raw))
        else:
            if action.size != 3:
                raise ValueError(f"Expected 3D action [ax, ay, tau_raw], got shape {action.shape}")
            a_xy = action[:2]
            tau_raw = float(action[2])

            ax_raw = float(a_xy[0])
            ay_raw = float(a_xy[1])

            raw_dv_dirmag = np.array([ax_raw, ay_raw], dtype=np.float64)
            raw_dv_dirmag_norm = float(np.linalg.norm(raw_dv_dirmag))

        dv_mag = 0.0
        burn_kind = "NONE"
        burn_applied = False
        

        extra_reward = 0.0
        extra_reward_terms: Dict[str, float] = {}

        # ---------- pre-TLI: exactly one TLI within window ----------
        used_tli_this_step = False

        if not self.tli_used:
                

            # Keep these only as diagnostics for now
            self.last_tli_u01_raw = float(np.clip(raw_dv_dirmag_norm, 0.0, 1.0))

            # -------------------------------------------------
            # V3_2: APPLY DV FIRST
            # -------------------------------------------------
            
            

            if self._use_tangential_scalar_action():
                dv_cmd_nominal = self._dv_vec_from_tangential_scalar(
                    ax_raw,
                    dv_cap=self._dv_cap_tli(),
                )
            else:
                dv_cmd_nominal = self._dv_vec_from_action_xy(
                    ax_raw,
                    ay_raw,
                    dv_cap=self._dv_cap_tli(),
                )
                dv_cmd_nominal = self._apply_pre_tli_control_mode(dv_cmd_nominal)
            dv_cmd = self._apply_burn_execution_noise(dv_cmd_nominal, burn_kind="TLI")
            dv_mag = float(np.linalg.norm(dv_cmd))

            # -----------------------------------------
            # PRE_TLI tiny-burn mask:
            # tiny random burns become exact zero
            # -----------------------------------------
            if dv_mag < self._pre_tli_burn_deadzone():
                dv_cmd[:] = 0.0
                dv_mag = 0.0

            self.last_tli_u01_exec = float(np.clip(dv_mag / max(self._dv_cap_tli(), 1e-12), 0.0, 1.0))

            # -------------------------------------------------
            # Pre-TLI burn logging only for now
            # We do NOT decide real TLI yet until after drift.
            # -------------------------------------------------
            if dv_mag > 0.0:
                self.dv_used += dv_mag
                self.state[2] += dv_cmd[0]
                self.state[3] += dv_cmd[1]
                self.pre_tli_last_burn_mag = float(dv_mag)

                if bool(self.cfg.staged_tli_enabled):
                    self.pre_tli_cum_dv += float(dv_mag)
                    self.pre_tli_burn_count += 1

                burn_kind = "PRE_TLI_BURN"
                burn_applied = True

                burn_event = {
                    "kind": "PRE_TLI_BURN",
                    "time": float(self.t),
                    "step_idx": int(self.step_idx),
                    "ax_raw": ax_raw,
                    "ay_raw": ay_raw,
                    "tau_raw": tau_raw,
                    "tau_true": np.nan,
                    "u01_raw": float(np.clip(raw_dv_dirmag_norm, 0.0, 1.0)),
                    "u01_exec": float(self.last_tli_u01_exec),
                    "pos_rot": self.state[:2].copy(),
                    "dv_vec_rot": np.array([dv_cmd[0], dv_cmd[1]], dtype=np.float64),
                    "dv_mag": float(dv_mag),
                    "obs_before_action": obs_before_action.copy(),
                    "pre_tli_cum_dv": float(getattr(self, "pre_tli_cum_dv", 0.0)),
                    "pre_tli_burn_count": int(getattr(self, "pre_tli_burn_count", 0)),
                }

                self.burn_events.append(burn_event)
                self.burns.append(np.array([dv_cmd[0], dv_cmd[1], dv_mag], dtype=np.float64))

            else:
                burn_kind = "PRE_TLI_WAIT"
                self.tli_did_burn = False
                self.pre_tli_last_burn_mag = 0.0

            # -------------------------------------------------
            # Drift after burn / wait
            # -------------------------------------------------
            dt_post = self._drift_time_from_tau_masked(tau_raw)

            if dt_post > 0.0:
                self._propagate(dt_post)
                self._consume_substep_events()

            self.last_dt_effective = float(dt_post)
            self.last_dt_warp = 0.0
            self.last_dt_post = float(dt_post)

            # -------------------------------------------------
            # Decide whether the real TLI has now happened
            # Trigger if either:
            #   1) burn magnitude >= 2.5 km/s
            #   2) Earth distance >= 0.1 nondim
            # whichever happens first
            # -------------------------------------------------
            rE_now, _ = dist_to_primaries(mu, self.state)

            dv_trigger = float(tli_ballistic_trigger_nondim())
            rE_trigger = float(tli_departure_trigger_radius())

            if not bool(self.cfg.staged_tli_enabled):
                # old behavior, unchanged
                is_real_tli = bool((dv_mag >= dv_trigger) or (rE_now >= rE_trigger))
            else:
                target_cum_dv = max(float(self.cfg.staged_tli_cumulative_dv_target), 1e-12)
                min_commit_frac = float(self.cfg.staged_tli_min_commit_frac_of_target)
                min_commit_dv = min_commit_frac * target_cum_dv

                hit_cumulative_target = bool(
                    bool(self.cfg.staged_tli_commit_on_cumulative_dv)
                    and (self.pre_tli_cum_dv >= target_cum_dv)
                )

                hit_departure_radius = bool(rE_now >= rE_trigger)

                hit_burn_count_guard = False
                if bool(self.cfg.staged_tli_limit_burn_count):
                    hit_burn_count_guard = bool(
                        (self.pre_tli_burn_count >= int(self.cfg.staged_tli_max_burn_count))
                        and (self.pre_tli_cum_dv >= min_commit_dv)
                    )

                is_real_tli = bool(
                    hit_cumulative_target
                    or hit_departure_radius
                    or hit_burn_count_guard
                )

            if is_real_tli and (not self.tli_ballistic_reward_given):
                used_tli_this_step = True
                if bool(self.cfg.staged_tli_enabled):
                    self.dv0 = float(self.pre_tli_cum_dv)
                else:
                    self.dv0 = float(dv_mag)
                self.tli_ax = ax_raw
                self.tli_ay = ay_raw
                self.tli_step_executed = int(self.step_idx)
                self.tli_tau = float(tau_raw)
                self.tli_theta = self._earth_centered_theta_of_state(self.state)
                self.tli_pos_rot = self.state[:2].copy()
                # Upgrade the last pre-TLI burn event to the actual commit marker
                if len(self.burn_events) > 0 and self.burn_events[-1]["kind"] == "PRE_TLI_BURN":
                    if bool(self.cfg.staged_tli_enabled):
                        self.burn_events[-1]["kind"] = "TLI_FINAL_BURN"
                        self.burn_events[-1]["pre_tli_cum_dv"] = float(self.pre_tli_cum_dv)
                        self.burn_events[-1]["pre_tli_burn_count"] = int(self.pre_tli_burn_count)
                        burn_kind = "TLI_FINAL_BURN"
                    else:
                        self.burn_events[-1]["kind"] = "TLI"
                        burn_kind = "TLI"
                elif dv_mag <= 0.0:
                    burn_kind = "TLI_WAIT_COMMIT"

                self.tli_state_after_burn = self.state.copy()

                if self.tli_state_after_burn is not None:
                    self._build_ballistic_reference_from_tli()
                
                self.tli_used = True
                self.tli_executed = True
                self.tli_ballistic_reward_given = True

                # Initialize real post-TLI invalid-orbit tracking
                rE_post_tli, _ = dist_to_primaries(mu, self.state)
                self.max_rE_seen_post_tli = float(rE_post_tli)
                self.invalid_return_armed_episode = bool(
                    rE_post_tli >= float(self.cfg.ballistic_invalid_return_arm_rE)
                )

                if self.cfg.reward_after_tli_ballistic_enabled:
                    dv_budget_now = float(self.reward_model.cfg.dv_budget) if self.reward_model is not None else np.inf

                    # If the REAL episode is already over budget, do NOT give any ballistic helper reward.
                    # The real env termination/reward should handle budget with the full penalty.
                    if self.dv_used > dv_budget_now:
                        self.ballistic_tli_reward_last = 0.0
                        self.ballistic_tli_min_rM_last = np.nan
                        self.ballistic_tli_min_rE_postflyby_last = np.nan
                        self.ballistic_tli_corridor_dist_last = np.nan
                        self.ballistic_tli_corridor_hit_last = False
                        self.ballistic_tli_success_last = False
                        self.ballistic_tli_vrel_at_min_rM_last = np.nan

                        extra_reward += 0.0
                        extra_reward_terms.update({
                            "r_tli_ballistic_total": 0.0,
                            "r_tli_ballistic_scale": 0.0,
                            "r_tli_ballistic_dv": 0.0,
                            "r_tli_ballistic_budget": 0.0,
                            "r_tli_ballistic_escape": 0.0,
                            "r_tli_ballistic_crash": 0.0,
                            "r_tli_ballistic_invalid_preflyby_earth_return": 0.0,
                            "r_tli_ballistic_flyby": 0.0,
                            "r_tli_ballistic_velocity": 0.0,
                            "r_tli_ballistic_return": 0.0,
                            "r_tli_ballistic_terminal_rewards_suppressed": 0.0,
                            "r_tli_ballistic_terminal_escape_flag": 0.0,
                            "r_tli_ballistic_terminal_timeout_flag": 0.0,
                            "r_tli_ballistic_terminal_success_flag": 0.0,
                            "r_tli_ballistic_term_reason_code": 0.0,
                            "r_tli_ballistic_skipped_due_to_real_budget_exceed": 1.0,
                        })
                    else:
                        ballistic_metrics = self._evaluate_ballistic_after_tli(
                            state_after_tli=self.tli_state_after_burn,
                            t_after_tli=float(self.t),
                        )
                        tli_ballistic_reward, tli_ballistic_terms = self._compute_ballistic_tli_reward(ballistic_metrics)

                        self.ballistic_tli_reward_last = float(tli_ballistic_reward)
                        self.ballistic_tli_min_rM_last = float(ballistic_metrics.get("min_rM", np.nan))
                        self.ballistic_tli_min_rE_postflyby_last = float(ballistic_metrics.get("min_rE_postflyby", np.nan))
                        self.ballistic_tli_corridor_dist_last = float(ballistic_metrics.get("corridor_dist", np.nan))
                        self.ballistic_tli_corridor_hit_last = bool(ballistic_metrics.get("corridor_hit_postflyby", False))
                        self.ballistic_tli_success_last = bool(ballistic_metrics.get("success", False))
                        self.ballistic_tli_vrel_at_min_rM_last = float(ballistic_metrics.get("vrel_at_min_rM", np.nan))

                        extra_reward += float(tli_ballistic_reward)
                        extra_reward_terms.update(tli_ballistic_terms)
            else:
                # Still pre-TLI, no ballistic reward yet
                self.tli_state_after_burn = None
                self.tli_used = False
                self.tli_executed = False
                self.ballistic_tli_reward_last = 0.0
                self.ballistic_tli_min_rM_last = np.nan
                self.ballistic_tli_min_rE_postflyby_last = np.nan
                self.ballistic_tli_corridor_dist_last = np.nan
                self.ballistic_tli_corridor_hit_last = False
                self.ballistic_tli_success_last = False
                self.ballistic_tli_vrel_at_min_rM_last = np.nan

        # ---------- post-TLI: free burn availability ----------
        else:
            if self.tli_used and self.cfg.mcc_enabled:
                if self._use_tangential_scalar_action():
                    # In PPO-A tangential-scalar mode, post-TLI burn control is disabled.
                    # This keeps the 2D PPO-A phasing experiment clean.
                    dv_cmd_nominal = np.zeros(2, dtype=np.float64)
                else:
                    dv_cmd_nominal = self._dv_vec_from_action_xy(
                        ax_raw, ay_raw, dv_cap=self._dv_cap_mcc()
                    )

                dv_cmd = self._apply_burn_execution_noise(dv_cmd_nominal, burn_kind="POST_TLI")
                dv_mag = float(np.linalg.norm(dv_cmd))

                if dv_mag > 0.0:
                    self.mcc_burn_count += 1
                    self.dv_mcc_total += dv_mag
                    self.dv_used += dv_mag
                    self.state[2] += dv_cmd[0]
                    self.state[3] += dv_cmd[1]

                    burn_kind = "POST_TLI"
                    burn_applied = True

                    self.burn_events.append({
                        "kind": "POST_TLI",
                        "time": float(self.t),
                        "step_idx": int(self.step_idx),
                        "ax_raw": ax_raw,
                        "ay_raw": ay_raw,
                        "tau_raw": tau_raw,
                        "tau_true": np.nan,
                        "u01_raw": float(np.clip(raw_dv_dirmag_norm, 0.0, 1.0)),
                        "u01_exec": float(np.clip(dv_mag / max(self._dv_cap_mcc(), 1e-12), 0.0, 1.0)),
                        "pos_rot": self.state[:2].copy(),
                        "dv_vec_rot": np.array([dv_cmd[0], dv_cmd[1]], dtype=np.float64),
                        "dv_mag": float(dv_mag),
                        "obs_before_action": obs_before_action.copy(),
                    })
                    self._maybe_store_mcc_ballistic_overlay(
                        state_after_burn=self.state.copy(),
                        t_after_burn=float(self.t),
                        burn_event=self.burn_events[-1],
                    )
                    self.burns.append(np.array([dv_cmd[0], dv_cmd[1], dv_mag], dtype=np.float64))

            dt_post = self._drift_time_from_tau_masked(tau_raw)

            if dt_post > 0.0:
                self._propagate(dt_post)
                self._consume_substep_events()

            self.last_dt_effective = float(dt_post)
            self.last_dt_warp = 0.0
            self.last_dt_post = float(dt_post)

        # ---------- update LEO gate AFTER propagation ----------
        rE_pos, _ = earth_moon_positions(mu)
        pos = self.state[:2].astype(np.float64)
        rE_now = float(np.linalg.norm(pos - rE_pos))

        if (not self.left_leo) and (rE_now >= self.r_leo_exit):
            self.left_leo = True
            self.left_leo_step = int(self.steps)
            self.left_leo_time = float(self.t)

        # ---------- step counters ----------
        self.step_idx += 1
        self.steps += 1


        # ---------- events ----------
        terminated, truncated, event_info = self._check_events()

        event_info = dict(event_info)
        event_info["left_leo"] = bool(self.left_leo)


        # ---------- force TLI-only termination AFTER reward ----------
        if used_tli_this_step and self.cfg.tli_only_mode:
            real_bad_terminal = str(event_info.get("term_reason", "")) in (
                "dv_budget_exceeded",
                "escape",
                "earth_impact",
                "moon_impact",
                "success",
            )

            terminated = True
            truncated = False

            if not real_bad_terminal:
                event_info["term_reason"] = "tli_only_done"
                event_info["success"] = bool(self.ballistic_tli_success_last)

        # ---------- base Sean reward ----------
        # IMPORTANT:
        # For TLI-only mode, do NOT let Sean terminal logic fire here.
        # We want only:
        #   per-step penalties + immediate ballistic TLI reward
        # then terminate the episode manually.
        reward_terminated = terminated
        reward_truncated = truncated

        # In TLI-only mode, suppress normal terminal reward only when the TLI step itself
        # is being ended artificially. Do NOT suppress real bad terminal events like
        # budget exceed, escape, or crashes.
        if used_tli_this_step and self.cfg.tli_only_mode:
            real_bad_terminal = str(event_info.get("term_reason", "")) in (
                "dv_budget_exceeded",
                "escape",
                "earth_impact",
                "moon_impact",
                "success",
            )
            if not real_bad_terminal:
                reward_terminated = False
                reward_truncated = False

        reward, reward_terms = self._compute_reward_sean(dv_mag, reward_terminated, reward_truncated, event_info)

        # ---------- add immediate TLI ballistic reward ----------
        reward += float(extra_reward)
        reward_terms = dict(reward_terms)
        reward_terms.update(extra_reward_terms)
        reward_terms["r_total"] = float(reward)
        reward_terms["r_ballistic_tli"] = float(extra_reward)

        # ---------- keep reward_record consistent with final merged reward ----------
        reward_record = reward_terms.get("reward_record", None)
        if isinstance(reward_record, dict):
            reward_record["step_reward"] = float(reward)

            if "terms" not in reward_record or not isinstance(reward_record["terms"], dict):
                reward_record["terms"] = {}

            # copy merged reward terms into the frozen record
            for k, v in reward_terms.items():
                if k == "reward_record":
                    continue
                if isinstance(v, (bool, int, float, np.floating)):
                    fv = float(v)
                    reward_record["terms"][str(k)] = fv if np.isfinite(fv) else 0.0
                else:
                    reward_record["terms"][str(k)] = 0.0

            reward_record["terms"]["r_total"] = float(reward)
            reward_record["terms"]["r_ballistic_tli"] = float(extra_reward)

            # optional but very helpful debug metrics
            if "metrics" not in reward_record or not isinstance(reward_record["metrics"], dict):
                reward_record["metrics"] = {}

            reward_record["metrics"]["extra_reward_total"] = float(extra_reward)
            reward_record["metrics"]["ballistic_tli_reward_last"] = float(self.ballistic_tli_reward_last)


        if burn_kind in ("TLI", "PRE_TLI_BURN", "TLI_WAIT_COMMIT", "PRE_TLI_WAIT"):
            u01_exec_log = float(self.last_tli_u01_exec)
        elif burn_kind == "POST_TLI":
            u01_exec_log = float(np.clip(dv_mag / max(self._dv_cap_mcc(), 1e-12), 0.0, 1.0))
        else:
            u01_exec_log = 0.0

        # ---------- outputs ----------
        obs = self._get_obs()
        info = self._get_info(extra={**event_info, **reward_terms})

        # ---------- store action log ----------
        self.action_history.append({
            "step_idx": int(self.step_idx - 1),
            "time_before": float(t_before_action),
            "time_after": float(self.t),
            "ax_raw": float(ax_raw),
            "ay_raw": float(ay_raw),
            "tau_raw": float(tau_raw),
            "tau_true_if_tli": (
                float(self.tli_tau)
                if burn_kind in ("TLI", "TLI_FINAL_BURN")
                else np.nan
            ),
            "u01_raw": float(np.clip(raw_dv_dirmag_norm, 0.0, 1.0)),
            "u01_exec": float(u01_exec_log),
            "burn_kind": str(burn_kind),
            "burn_applied": bool(burn_applied),
            "dv_mag": float(dv_mag),
            "dt_effective": float(self.last_dt_effective),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "state_before_action": np.asarray(state_before_action, dtype=np.float64).copy(),
            "state_after_step": np.asarray(self.state, dtype=np.float64).copy(),
            "obs_before_action": np.asarray(obs_before_action, dtype=np.float32).copy(),
            "obs_after_step": np.asarray(obs, dtype=np.float32).copy(),
            "info_selected": {
                "rE": float(info.get("rE", np.nan)),
                "rM": float(info.get("rM", np.nan)),
                "dv_used": float(info.get("dv_used", np.nan)),
                "term_reason": str(info.get("term_reason", "")),
                "flyby_done": bool(info.get("flyby_done", False)),
                "return_corridor_hit_postflyby": bool(info.get("return_corridor_hit_postflyby", False)),
                "ballistic_tli_corridor_hit": bool(info.get("ballistic_tli_corridor_hit", False)),
                "left_leo": bool(info.get("left_leo", False)),
            },
        })

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------------

    def _check_invalid_post_tli_event(self) -> Optional[str]:
        """
        Real-episode invalid-orbit detection.
        Active only after committed TLI, only before flyby, and only when not in tli_only_mode.
        """
        if not bool(self.cfg.ballistic_invalid_preflyby_return_enabled):
            return None

        if bool(self.cfg.tli_only_mode):
            return None

        if not bool(self.tli_used):
            return None

        if bool(self.flyby_done):
            return None

        rE_pos, rM_pos = earth_moon_positions(self.cfg.mu)
        pos = self.state[:2].astype(np.float64)
        vel = self.state[2:4].astype(np.float64)

        rE = float(np.linalg.norm(pos - rE_pos))
        rM = float(np.linalg.norm(pos - rM_pos))

        self.max_rE_seen_post_tli = max(float(self.max_rE_seen_post_tli), rE)

        if rE >= float(self.cfg.ballistic_invalid_return_arm_rE):
            self.invalid_return_armed_episode = True

        # Earth-centered inertial radial velocity
        v_sc_I = v_rot_to_inertial(pos, vel)
        v_earth_I = omega_cross_r_2d(rE_pos)
        r_sc_E = pos - rE_pos
        v_sc_E_I = v_sc_I - v_earth_I
        vrE = radial_velocity_about_point(r_sc_E, v_sc_E_I)

        # Case 1: obvious Earth-bound junk after TLI.
        # The trajectory never got meaningfully away from Earth,
        # is still far from the Moon,
        # and is no longer clearly on an outbound transfer.
        if (
            self.mcc_burn_count >= 1
            and self.max_rE_seen_post_tli < float(self.cfg.ballistic_invalid_stuck_max_rE)
            and rM > float(self.cfg.ballistic_invalid_return_moon_far_rM)
            and (
                rE < 0.9 * float(self.cfg.ballistic_invalid_stuck_max_rE)
                or vrE <= 0.0
            )
        ):
            return "invalid_preflyby_earth_return"

        # Case 2: got outbound, now clearly falling back to Earth while still far from Moon.
        if (
            bool(self.invalid_return_armed_episode)
            and (vrE <= float(self.cfg.ballistic_invalid_return_vrE_threshold))
            and (rM > float(self.cfg.ballistic_invalid_return_moon_far_rM))
        ):
            return "invalid_preflyby_earth_return"

        return None

    def _check_events(self):
        term_reason = ""

        rE, rM = dist_to_primaries(self.cfg.mu, self.state)
        pos = self.state[:2]
        r_norm = float(np.linalg.norm(pos))

        terminated = False
        truncated = False

        if self._early_terminate is not None:
            term_reason = str(self._early_terminate[0])
            terminated = True

        elif r_norm > float(self.cfg.r_escape):
            term_reason = "escape"
            terminated = True

        elif self.cfg.terminate_on_dv_budget_exceed and (self.dv_used > self.reward_model.cfg.dv_budget):
            term_reason = "dv_budget_exceeded"
            terminated = True

        elif self.success:
            term_reason = "success"
            terminated = True

        # still not even at the configured departure ring
        elif self.cfg.terminate_if_no_leo_exit and self._pre_tli_mode() and (not self.left_leo):
            if self.t >= float(self.pre_tli_timeout_limit):
                term_reason = "no_tli_3_orbits"
                terminated = True
                self.no_tli_terminated = True
                self.no_tli_termination_time = float(self.t)

        # crossed departure ring, but still no real TLI
        elif self._pre_tli_mode() and self.left_leo:
            grace = float(self._left_leo_no_tli_grace_nondim())
            t_left = float(self.left_leo_time) if np.isfinite(self.left_leo_time) else float(self.t)
            if self.t >= t_left + grace:
                term_reason = "left_leo_no_tli"
                terminated = True
        
        elif (invalid_reason := self._check_invalid_post_tli_event()) is not None:
            term_reason = invalid_reason
            terminated = True

        elif self.t >= float(self.cfg.t_max):
            term_reason = "timeout"
            truncated = True

        if terminated or truncated:
            self.terminal_marker_rot = self.state[:2].copy()
        else:
            self.terminal_marker_rot = None

        info = self._get_info(extra={"term_reason": term_reason})
        return terminated, truncated, info

    # ------------------------------------------------------------
    # BASE REWARD
    # ------------------------------------------------------------
    def _compute_reward_sean(
        self,
        dv_mag: float,
        terminated: bool,
        trunc: bool,
        event_info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        if self.reward_model is None:
            return 0.0, {
                "r_total": 0.0,
                "reward_record": {
                    "step_reward": 0.0,
                    "terms": {"r_total": 0.0},
                    "flags": {},
                    "metrics": {},
                    "config": {},
                },
            }

        full_info = self._get_info(extra=event_info)
        r, terms = self.reward_model.compute(
            env=self,
            state=self.state,
            info=full_info,
            dv_mag=float(dv_mag),
            terminated=terminated,
            truncated=trunc,
        )

        terms = dict(terms)
        terms["r_total"] = float(r)

        # Freeze the exact reward-time snapshot used for this step.
        reward_record = {
            "step_reward": float(r),
            "terms": {},
            "flags": {},
            "metrics": {},
            "config": {},
        }

        # Copy all numeric reward/flag terms exactly as produced by reward_model.compute()
        for k, v in terms.items():
            if isinstance(v, (bool, int, float, np.floating)):
                fv = float(v)
                reward_record["terms"][str(k)] = fv if np.isfinite(fv) else 0.0
            else:
                reward_record["terms"][str(k)] = 0.0

        # Runtime flags that matter for reward gating / interpretation
        reward_record["flags"] = {
            "terminated": bool(terminated),
            "truncated": bool(trunc),
            "success": bool(getattr(self, "success", False)),
            "flyby_done": bool(getattr(self, "flyby_done", False)),
            "return_done": bool(getattr(self, "return_done", False)),
            "left_leo": bool(getattr(self, "left_leo", False)),
            "tli_used": bool(getattr(self, "tli_used", False)),
            "tli_executed": bool(getattr(self, "tli_executed", False)),
            "tli_ballistic_reward_given": bool(getattr(self, "tli_ballistic_reward_given", False)),
        }

        # Geometry / state metrics actually relevant to reward interpretation
        reward_record["metrics"] = {
            "time": float(getattr(self, "t", np.nan)),
            "dv_mag_step": float(dv_mag),
            "dv_used_total": float(getattr(self, "dv_used", np.nan)),
            "min_rM_env": float(getattr(self, "min_rM", np.nan)),
            "min_rE_env": float(getattr(self, "min_rE", np.nan)),
            "min_rE_postflyby_env": float(getattr(self, "min_rE_postflyby", np.nan)),
            "ballistic_tli_reward_last": float(getattr(self, "ballistic_tli_reward_last", np.nan)),
            "rE_info": float(full_info.get("rE", np.nan)),
            "rM_info": float(full_info.get("rM", np.nan)),
            "vrel_moon_info": float(full_info.get("vrel_moon", np.nan)),
            "term_reason": str(full_info.get("term_reason", "")),
            "reward_model_min_rM": float(getattr(self.reward_model, "min_rM", np.nan)),
            "reward_model_v_at_min_rM": float(getattr(self.reward_model, "v_at_min_rM", np.nan)),
        }

        # Active config snapshot at reward time
        reward_record["config"] = {
            "cfg_r_moon_flyby": float(getattr(self.cfg, "r_moon_flyby", np.nan)),
            "cfg_rp_min": float(getattr(self.cfg, "rp_min", np.nan)),
            "cfg_rp_max": float(getattr(self.cfg, "rp_max", np.nan)),
            "cfg_tli_only_mode": bool(getattr(self.cfg, "tli_only_mode", False)),
            "cfg_reward_after_tli_ballistic_enabled": bool(getattr(self.cfg, "reward_after_tli_ballistic_enabled", False)),
            "cfg_dv_noise_sigma_tli": float(getattr(self.cfg, "dv_noise_sigma_tli", np.nan)),
            "cfg_dv_noise_sigma_mcc": float(getattr(self.cfg, "dv_noise_sigma_mcc", np.nan)),
            "w_flyby": float(getattr(self.reward_model.w, "w_flyby", np.nan)),
            "w_velocity": float(getattr(self.reward_model.w, "w_velocity", np.nan)),
            "w_dv": float(getattr(self.reward_model.w, "w_dv", np.nan)),
            "w_return": float(getattr(self.reward_model.w, "w_return", np.nan)),
            "w_budget": float(getattr(self.reward_model.w, "w_budget", np.nan)),
            "w_escape": float(getattr(self.reward_model.w, "w_escape", np.nan)),
            "w_earth_crash": float(getattr(self.reward_model.w, "w_earth_crash", np.nan)),
            "w_moon_crash": float(getattr(self.reward_model.w, "w_moon_crash", np.nan)),
            "w_postflyby_earth_crash": float(getattr(self.reward_model.w, "w_postflyby_earth_crash", np.nan)),
            "w_invalid_preflyby_earth_return": float(getattr(self.reward_model.w, "w_invalid_preflyby_earth_return", np.nan)),
            "flyby_reward_gate": float(getattr(self.reward_model.cfg, "flyby_reward_gate", np.nan)),
            "dv_budget": float(getattr(self.reward_model.cfg, "dv_budget", np.nan)),
            "v_target_moon": float(getattr(self.reward_model.cfg, "v_target_moon", np.nan)),
            "v_deadzone": float(getattr(self.reward_model.cfg, "v_deadzone", np.nan)),
        }

        # Keep current flat keys for backward compatibility with existing plotting/report code.
        terms["reward_record"] = reward_record
        return float(r), terms

    # ------------------------------------------------------------
    # OBS + INFO
    # ------------------------------------------------------------

    
        
    def _get_obs(self):
        x, y, vx, vy = self.state
        rE, rM = dist_to_primaries(self.cfg.mu, self.state)
        C = jacobi_constant(self.cfg.mu, self.state)
        dv_budget = max(self.reward_model.cfg.dv_budget, 1e-12)

        obs = [
            float(x / self.cfg.pos_scale),
            float(y / self.cfg.pos_scale),
            float(vx / self.cfg.vel_scale),
            float(vy / self.cfg.vel_scale),
            float(rE / self.cfg.pos_scale),
            float(rM / self.cfg.pos_scale),
            float(C / self.cfg.c_scale),
            float(self.t / max(self.cfg.t_max, 1e-12)),
            float(np.clip(self.dv_used / dv_budget, 0.0, 2.0)),
        ]

        if self.cfg.add_phase_angle_obs:
            phase = phase_angle_sc_vs_moon_about_earth(self.cfg.mu, self.state)
            obs.append(float(phase / np.pi))

        if self.cfg.add_mode_obs and self.cfg.add_legacy_mode_obs:
            _, tau_max_nd = self._current_tau_bounds_nondim()
            tau_max_global_nd = minutes_to_nondim_time(
                max(
                    float(RUN.drift_max_minutes_pre_tli),
                    float(RUN.drift_max_minutes_post_tli),
                )
            )
            dv_cap_now = self._current_dv_cap()
            dv_cap_ref = max(float(self._dv_cap_tli()), 1e-12)

            pre_tli_clock = 0.0
            if self._pre_tli_mode():
                pre_tli_clock = float(self.t / max(self.pre_tli_timeout_limit, 1e-12))
                pre_tli_clock = float(np.clip(pre_tli_clock, 0.0, 1.0))

            obs.extend([
                float(1.0 if self.tli_used else 0.0),
                float(tau_max_nd / max(tau_max_global_nd, 1e-12)),
                float(dv_cap_now / max(dv_cap_ref, 1e-12)),
                float(pre_tli_clock),
            ])
        if self.cfg.add_mode_obs and self.cfg.add_staged_tli_obs and self.cfg.staged_tli_enabled:
            target = max(float(self.cfg.staged_tli_cumulative_dv_target), 1e-12)
            cum_dv_norm = float(np.clip(self.pre_tli_cum_dv / target, 0.0, 2.0))

            max_count = max(int(self.cfg.staged_tli_max_burn_count), 1)
            burn_count_norm = float(np.clip(self.pre_tli_burn_count / max_count, 0.0, 2.0))

            obs.extend([
                cum_dv_norm,
                burn_count_norm,
            ])

        return np.asarray(obs, dtype=np.float32)

    def _get_info(self, extra: Optional[Dict[str, Any]] = None):
        rE, rM = dist_to_primaries(self.cfg.mu, self.state)

        info = {
            "t": float(self.t),
            "rE": float(rE),
            "rM": float(rM),
            "dv_used": float(self.dv_used),
            "dv0": float(self.dv0),
            "dv_mcc_total": float(self.dv_mcc_total),
            "left_leo": bool(self.left_leo),
            "flyby_done": bool(self.flyby_done),
            "return_done": bool(self.return_done),
            "success": bool(self.success),
            "min_rM": float(self.min_rM),
            "min_rE": float(self.min_rE),
            "min_rE_postflyby": float(self.min_rE_postflyby),
            "best_postflyby_corridor_dist": float(self.best_postflyby_corridor_dist),
            "return_corridor_hit_postflyby": bool(self.return_corridor_hit_postflyby),

            "dt_effective_last": float(self.last_dt_effective),
            "dt_warp_last": float(self.last_dt_warp),
            "dt_post_last": float(self.last_dt_post),

            "spawn_theta": float(self.spawn_theta) if np.isfinite(self.spawn_theta) else np.nan,
            "tli_theta": float(self.tli_theta) if np.isfinite(self.tli_theta) else np.nan,

            "ballistic_tli_reward": float(self.ballistic_tli_reward_last),
            "ballistic_tli_min_rM": float(self.ballistic_tli_min_rM_last),
            "ballistic_tli_min_rE_post": float(self.ballistic_tli_min_rE_postflyby_last),
            "ballistic_tli_corridor_dist": float(self.ballistic_tli_corridor_dist_last),
            "ballistic_tli_corridor_hit": bool(self.ballistic_tli_corridor_hit_last),

            "cfg_tli_only_mode": bool(self.cfg.tli_only_mode),
            "cfg_reward_after_tli": bool(self.cfg.reward_after_tli_ballistic_enabled),
            "cfg_dv_noise_sigma_tli": float(self.cfg.dv_noise_sigma_tli),
            "cfg_dv_noise_sigma_mcc": float(self.cfg.dv_noise_sigma_mcc),

            "pre_tli_cum_dv": float(getattr(self, "pre_tli_cum_dv", 0.0)),
            "pre_tli_burn_count": int(getattr(self, "pre_tli_burn_count", 0)),
            "pre_tli_last_burn_mag": float(getattr(self, "pre_tli_last_burn_mag", 0.0)),
            "staged_tli_enabled": bool(getattr(self.cfg, "staged_tli_enabled", False)),

            "mode_pre_tli": bool(self._pre_tli_mode()),
            "current_dv_cap": float(self._current_dv_cap()),
            "current_tau_min_minutes": float(self._current_tau_bounds_minutes()[0]),
            "current_tau_max_minutes": float(self._current_tau_bounds_minutes()[1]),
            "pre_tli_timeout_limit": float(self.pre_tli_timeout_limit),
            "left_leo_trigger_rE": float(self.r_leo_exit),
            "left_leo_time": float(self.left_leo_time) if np.isfinite(self.left_leo_time) else np.nan,
            "left_leo_no_tli_grace": float(self._left_leo_no_tli_grace_nondim()),
            "no_tli_terminated": bool(self.no_tli_terminated),
            "left_leo_step": float(self.left_leo_step) if self.left_leo_step is not None else np.nan,
            "tli_tau": float(self.tli_tau) if np.isfinite(self.tli_tau) else np.nan,
            "tli_ax": float(self.tli_ax) if np.isfinite(self.tli_ax) else np.nan,
            "tli_ay": float(self.tli_ay) if np.isfinite(self.tli_ay) else np.nan,
            "tli_step_executed": int(self.tli_step_executed),
            "tli_used": bool(self.tli_used),
            "tli_executed": bool(self.tli_executed),
            "tli_ballistic_reward_given": bool(self.tli_ballistic_reward_given),
            "mcc_ballistic_overlay_count": int(len(self.mcc_ballistic_overlays)),
            "cfg_trainer_mode": str(self.cfg.trainer_mode),
            "cfg_tli_control_mode": str(self.cfg.tli_control_mode),
            "cfg_ppo_b_baseline_theta": float(self.cfg.ppo_b_baseline_theta),
            "cfg_ppo_b_baseline_ax": float(self.cfg.ppo_b_baseline_ax),
            "cfg_ppo_b_baseline_ay": float(self.cfg.ppo_b_baseline_ay),
            "cfg_ppo_b_baseline_tau": float(self.cfg.ppo_b_baseline_tau),
            "ppo_b_scenario_index": int(getattr(self, "ppo_b_scenario_index", -1)),
            "ppo_b_scenario_label": int(getattr(self, "ppo_b_scenario_label", -1)),
            "ppo_b_scenario_ballistic_min_rM": float(getattr(self, "ppo_b_scenario_ballistic_min_rM", np.nan)),
            "ppo_b_scenario_ballistic_corridor_dist": float(getattr(self, "ppo_b_scenario_ballistic_corridor_dist", np.nan)),
            "ppo_b_scenario_ballistic_success": bool(getattr(self, "ppo_b_scenario_ballistic_success", False)),

            "ppo_b_scenario_index": int(getattr(self, "ppo_b_scenario_index", -1)),
            "ppo_b_scenario_row_index": int(getattr(self, "ppo_b_scenario_row_index", -1)),
            "ppo_b_scenario_label": int(getattr(self, "ppo_b_scenario_label", -1)),
            "ppo_b_scenario_term_reason": str(getattr(self, "ppo_b_scenario_term_reason", "")),
            "ppo_b_scenario_ballistic_min_rM": float(getattr(self, "ppo_b_scenario_ballistic_min_rM", np.nan)),
            "ppo_b_scenario_ballistic_corridor_dist": float(getattr(self, "ppo_b_scenario_ballistic_corridor_dist", np.nan)),
            "ppo_b_scenario_ballistic_success": bool(getattr(self, "ppo_b_scenario_ballistic_success", False)),
            "ppo_b_scenario_fixed_index_mode": bool(getattr(self, "ppo_b_scenario_fixed_index_mode", False)),
            "ppo_b_scenario_noise_pos_sigma": float(getattr(self, "ppo_b_scenario_noise_pos_sigma", 0.0)),
            "ppo_b_scenario_noise_vel_sigma": float(getattr(self, "ppo_b_scenario_noise_vel_sigma", 0.0)),

            "cfg_ppo_b_case_source": str(getattr(self.cfg, "ppo_b_case_source", "")),
            "cfg_ppo_b_library_path": str(getattr(self.cfg, "ppo_b_library_path", "")),
            "cfg_ppo_b_prob_good": float(getattr(self.cfg, "ppo_b_prob_good", 0.0)),
            "cfg_ppo_b_prob_savable": float(getattr(self.cfg, "ppo_b_prob_savable", 0.0)),
            "cfg_ppo_b_prob_bad": float(getattr(self.cfg, "ppo_b_prob_bad", 0.0)),
            "cfg_ppo_b_noise_theta_deg": float(getattr(self.cfg, "ppo_b_noise_theta_deg", 0.0)),
            "cfg_ppo_b_noise_tli_dir_deg": float(getattr(self.cfg, "ppo_b_noise_tli_dir_deg", 0.0)),
            "cfg_ppo_b_noise_tli_dv_kms": float(getattr(self.cfg, "ppo_b_noise_tli_dv_kms", 0.0)),
        }
        info["dt_ratio"] = float(max(0.0, self.last_dt_effective / max(self.cfg.dt, 1e-12)))

        if extra is not None:
            info.update(extra)
        return info