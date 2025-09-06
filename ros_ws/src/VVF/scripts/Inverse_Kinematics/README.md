# Inverse Kinematics Tools (Kinova Gen3 Lite – 3-DOF subset)

Three scripts to compute, test, and visualize inverse kinematics (IK) for a simplified 3-joint model of the Kinova Gen3 Lite.  
Units: **meters** (links) and **radians** (angles).

---

## Repository

    inverse_kinematics.py            # Reusable FK/IK functions
    inverse_kinematics_calc.py       # CLI: solve IK for a single pose
    kinova_gen3lite_ik_analyzer.py   # Workspace sampling + plots + CSV

---

## Install

- Python 3.8+
- Packages (for plotting/analysis):

    pip install numpy matplotlib

---

## Model constants

    d1=0.057  d2=0.48  d3=0.28  d4=0.01  d5=0.4273  d6=0.12
    q1 ∈ [-2.68, 2.68],  q2 ∈ [-2.61, 2.61],  q3 ∈ [-2.61, 2.61]
    tolerance (FK↔IK match) = 1e-3

**Forward kinematics**

    XE = d1 cos(q2−q3) cos q1 − d2 sin(q2−q3) cos q1 − d3 sin q2 cos q1 + d4 sin q1 + d6
    YE = d1 cos(q2−q3) sin q1 − d2 sin(q2−q3) sin q1 − d3 sin q2 sin q1 − d4 cos q1
    ZE = d1 sin(q2−q3) + d2 cos(q2−q3) + d3 cos q2 + d5

---

## What each script does

### `inverse_kinematics.py`
- Public functions:
  - `calculate_forward_kinematics(q1, q2, q3) -> (XE, YE, ZE)`
  - `inverse_kinematics(XE, YE, ZE) -> list[(q1, q2, q3)]`
- Internals:
  - Closed-form steps: `inverse_for_q1`, `inverse_for_q23`, `inverse_for_q2` (via `acos`), `inverse_for_q3`
  - Angle normalization + joint-limit filtering
  - Deduplicates near-identical solutions (≈1e-2)
  - Guards a small singular tube around the wrist axis:  
    `(XE − d6)^2 + YE^2 < 1e-3 → no solution`

### `inverse_kinematics_calc.py`
- Self-contained CLI wrapper; echoes the target and prints each valid `(q1, q2, q3)`.

### `kinova_gen3lite_ik_analyzer.py`
- Loops a grid over `q1,q2,q3` (default 25 steps each)
- For each FK pose, runs IK and counts unique solutions (0/1/2/4)
- Plots per-category 3D clouds and writes a CSV

---

## Quick start

### Use as a library (`inverse_kinematics.py`)

    from inverse_kinematics import inverse_kinematics, calculate_forward_kinematics
    XE, YE, ZE = 0.30, 0.10, 0.55
    solutions = inverse_kinematics(XE, YE, ZE)  # list of (q1, q2, q3)
    print(solutions)
    # forward check (first solution)
    print(calculate_forward_kinematics(*solutions[0]))

### CLI solver (`inverse_kinematics_calc.py`)

    python inverse_kinematics_calc.py XE YE ZE
    # example:
    python inverse_kinematics_calc.py 0.30 0.10 0.55

### Workspace analyzer (`kinova_gen3lite_ik_analyzer.py`)

    python kinova_gen3lite_ik_analyzer.py

Outputs:
- Console stats for counts of **0/1/2/4** solutions
- 3D scatter plots (one per category)
- `position_data_with_solutions.csv` with columns: `XE,YE,ZE,Num_Solutions`

---

## Tips & tweaks

- Speed vs. detail: change analyzer grid size (`np.linspace(..., 25)`)
- Numerics: if a real solution is missed, try `tolerance = 3e-3`
- Headless: comment out plotting; CSV still saves
- Units & limits: all angles are radians; functions enforce joint ranges

---

## Troubleshooting

- **“Invalid input: XE and YE out of bounds”** (CLI)  
  Target lies in the forbidden tube near the wrist axis; adjust `XE, YE` slightly.

- **No solutions when expected**  
  Increase `tolerance` a bit and confirm pose reachability with FK.
