from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time

from config import RUN
from cr3bp_env_v4 import (
    rk4_step,
    kms_to_nondim_dv,
    minutes_to_nondim_time,
)

print("="*70)
print("GLOBAL MCC IMPULSE OPTIMIZER")
print("="*70)

print("[1] Rough library")
print("[2] Custom library")
print("[3] Latest staged handoff")

choice=input("Selection: ")

from pathlib import Path
import glob

script_dir = Path(__file__).resolve().parent

if choice=="1":

    candidates=list(
        script_dir.glob(
            "rough_scenario_classification/*.npz"
        )
    )

    if len(candidates)==0:
        raise FileNotFoundError(
            "\nNo .npz files found in:\n"
            f"{script_dir/'rough_scenario_classification'}"
        )

    print("\nAvailable handoff libraries:\n")

    for i,f in enumerate(candidates):
        print(f"[{i}] {f.name}")

    k=int(input("\nSelection: "))

    lib=str(candidates[k])

elif choice=="2":

    lib=input(
        "Full path to library: "
    )

else:

    candidates=sorted(
        script_dir.glob(
            "rough_scenario_classification/*staged*.npz"
        ),
        key=lambda x:x.stat().st_mtime,
        reverse=True
    )

    if len(candidates)==0:
        raise FileNotFoundError(
            "No staged handoff files found."
        )

    lib=str(
        candidates[0]
    )

print("\nUsing:\n")
print(lib)


data=np.load(lib,allow_pickle=True)

states=data["state_handoff"]

print()
print(f"Found {len(states)} trajectories")
idx=int(input("Trajectory index: "))

x0=states[idx].copy()

print()
print("[1] Quick (~20 min)")
print("[2] Medium (~2h)")
print("[3] Overnight (~6–8h)")

mode=input("Selection: ")

if mode=="1":
    popsize=10
    maxiter=40

elif mode=="2":
    popsize=20
    maxiter=100

else:
    popsize=35
    maxiter=250


traj_best=None



dt=0.0005

def cr3bp_rhs(s):
    x,y,vx,vy=s
    mu=0.0121505856

    r1=((x+mu)**2+y**2)**0.5
    r2=((x-(1-mu))**2+y**2)**0.5

    ax=(
        x
        +2*vy
        -(1-mu)*(x+mu)/(r1**3)
        -mu*(x-(1-mu))/(r2**3)
    )

    ay=(
        y
        -2*vx
        -(1-mu)*y/(r1**3)
        -mu*y/(r2**3)
    )

    return np.array([vx,vy,ax,ay])


def evaluate(X):

    global traj_best

    dv=float(X[0])
    theta=np.deg2rad(float(X[1]))

    state=x0.copy().astype(float)

    dv_nd=kms_to_nondim_dv(dv/1000.0)

    state[2]+=dv_nd*np.cos(theta)
    state[3]+=dv_nd*np.sin(theta)

    traj=[]

    moon_hit=False
    earth_hit=False

    entered_lunar_flyby=False
    exited_lunar_flyby_after_entry=False
    passed_lunar_periapsis=False

    min_rM=1e9
    prev_rM=None
    rM_increasing_count=0

    min_rE_postflyby=1e9

    r_moon_flyby=0.05

    r_return_low=0.020
    r_return_high=0.050

    r_earth_impact=0.017
    r_moon_impact=0.0045

    t=0.0

    while t<4.0:

        traj.append(state.copy())

        k1=cr3bp_rhs(state)
        k2=cr3bp_rhs(state+0.5*dt*k1)
        k3=cr3bp_rhs(state+0.5*dt*k2)
        k4=cr3bp_rhs(state+dt*k3)

        state=state+(dt/6.0)*(k1+2*k2+2*k3+k4)

        x,y,vx,vy=state

        mu=0.0121505856
        earth=np.array([-mu,0.0])
        moon=np.array([1.0-mu,0.0])

        rE=np.linalg.norm(np.array([x,y])-earth)
        rM=np.linalg.norm(np.array([x,y])-moon)

        min_rM=min(min_rM,rM)

        # Must ENTER lunar flyby radius first
        if rM <= r_moon_flyby:
            entered_lunar_flyby=True

        # Lunar periapsis detection:
        # after entering the lunar flyby sphere, rM must start increasing
        if entered_lunar_flyby and prev_rM is not None:
            if rM > prev_rM:
                rM_increasing_count+=1
            else:
                rM_increasing_count=0

            if rM_increasing_count >= 3:
                passed_lunar_periapsis=True

        # Must EXIT lunar flyby radius after having entered it
        if entered_lunar_flyby and passed_lunar_periapsis and rM > r_moon_flyby:
            exited_lunar_flyby_after_entry=True

        # Only count Earth return AFTER lunar flyby was completed
        if exited_lunar_flyby_after_entry:
            min_rE_postflyby=min(min_rE_postflyby,rE)

        prev_rM=rM

        # Impacts
        if rM < r_moon_impact:
            moon_hit=True
            break

        if rE < r_earth_impact:
            earth_hit=True
            break

        t+=dt

    valid_lunar_flyby=bool(
        entered_lunar_flyby
        and passed_lunar_periapsis
        and exited_lunar_flyby_after_entry
    )

    return_corridor_hit=bool(
        valid_lunar_flyby
        and (r_return_low <= min_rE_postflyby <= r_return_high)
    )

    # ------------------------------------------------------------
    # Objective:
    # Minimize DV ONLY for candidates that:
    #   1) enter lunar flyby radius
    #   2) pass lunar periapsis
    #   3) exit lunar flyby radius
    #   4) enter Earth return corridor after flyby
    # ------------------------------------------------------------

    if return_corridor_hit and not earth_hit and not moon_hit:
        score=dv

    else:
        score=1e6

        # Strong discrete penalties for wrong event order
        if not entered_lunar_flyby:
            score+=300000.0

        if entered_lunar_flyby and not passed_lunar_periapsis:
            score+=200000.0

        if passed_lunar_periapsis and not exited_lunar_flyby_after_entry:
            score+=150000.0

        if valid_lunar_flyby and not return_corridor_hit:
            score+=100000.0

        if earth_hit:
            score+=500000.0

        if moon_hit:
            score+=500000.0

        # Continuous shaping toward lunar flyby radius
        if not valid_lunar_flyby:
            score+=10000.0*max(0.0,min_rM-r_moon_flyby)

        # Continuous shaping toward return corridor, only after valid lunar flyby
        if valid_lunar_flyby:
            if min_rE_postflyby < r_return_low:
                corridor_error=r_return_low-min_rE_postflyby
            elif min_rE_postflyby > r_return_high:
                corridor_error=min_rE_postflyby-r_return_high
            else:
                corridor_error=0.0

            score+=50000.0*corridor_error

        # Very small DV term, only as tie-breaker for invalid candidates
        score+=0.01*dv

    if score<evaluate.best:
        evaluate.best=score
        traj_best=np.array(traj)

        print(
            f"\nBEST:"
            f" dv={dv:.2f}"
            f" angle={np.rad2deg(theta):.1f}"
            f" score={score:.2f}"
            f" flyby={valid_lunar_flyby}"
            f" return={return_corridor_hit}"
            f" min_rM={min_rM:.5f}"
            f" min_rE_post={min_rE_postflyby:.5f}"
        )

    return float(score)

evaluate.best=1e9


start=time.time()

result=differential_evolution(
    evaluate,
    bounds=[
        (0,60),
        (0,360)
    ],
    popsize=popsize,
    maxiter=maxiter,
    polish=True
)

elapsed=(
    time.time()-start
)/3600

print("\n")
print("="*70)
print("BEST SOLUTION")
print("="*70)

print(
f"DV: {result.x[0]:.3f} m/s"
)

print(
f"Angle: {result.x[1]:.3f} deg"
)

print(
f"Runtime: {elapsed:.2f} h"
)

plt.figure()

plt.plot(
    traj_best[:,0],
    traj_best[:,1]
)

mu=0.0121505856

plt.scatter(
    -mu,
    0
)

plt.scatter(
    1-mu,
    0
)

plt.axis("equal")

import json
from datetime import datetime

stamp=datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

save_dir=Path(
    f"Saved Policies/global_mcc_{stamp}"
)

save_dir.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    save_dir/"best_trajectory.png",
    dpi=300
)

np.savez(
    save_dir/"best_trajectory_data.npz",
    trajectory=traj_best,
    dv=result.x[0],
    angle=result.x[1]
)

with open(
    save_dir/"best_solution.json",
    "w"
) as f:

    json.dump(
        {
            "dv_mps":float(result.x[0]),
            "angle_deg":float(result.x[1]),
            "runtime_hr":float(elapsed)
        },
        f,
        indent=4
    )

plt.show()