# ROS Nodes: `back_to_home.py`, `dynamic_obstacle.py` & `kinematics_validation.py`

Three minimal ROS (Python) nodes:
- **back_to_home.py** – drives a 6-joint arm to a fixed “home” pose via velocity control.
- **dynamic_obstacle.py** – moves a Gazebo model back-and-forth along **Y**.
- **kinematics_validation.py** – validates FK/Jacobian vs. real TF; logs errors to CSV and publishes small test velocities.

---

## Files
- back_to_home.py
- dynamic_obstacle.py
- kinematics_validation.py

## Dependencies
- ROS (rospy, tf)
- Topics / services  
  - /joint_states  (sensor_msgs/JointState)  
  - /kinova_arm_joint_{1..6}_velocity_controller/command  (std_msgs/Float64)  
  - /gazebo/set_model_state  (gazebo_msgs/SetModelState)  
- TF frames: `base_link` ↔ `kinova_arm_tool_frame`
- Python: `numpy`, `csv`

Install (Python):
    
    pip install numpy

---

## 1) back_to_home.py
**Logic**
- Subscribes `/joint_states`; computes velocity = `kp*(goal − q)`; caps speed; publishes to 6 velocity controllers.
- Defaults: `goal = [0.7, −0.8, 1.1, 0, 0, 0]` rad, `kp = 1.0`, `vmax = 0.15` rad/s, stop when `|goal − q| < 0.01`, rate = 10 Hz.
- Joint indexing assumes positions at indices **2..7** in `JointState`; adjust if yours differ.

Run:
    
    rosrun VVF back_to_home.py

Edit in file if needed: `joints_goal`, `kp`, `vmax`, threshold, joint indices.

---

## 2) dynamic_obstacle.py
**Logic**
- Oscillates an existing Gazebo model along **Y** between `start_y` and `end_y` at `velocity`, at `update_rate` Hz.
- Waits for `/gazebo/set_model_state` and updates pose each cycle.

Parameters (private; set with `_param:=value`):
- `~model_name` (string, required)
- `~type` (string, optional)
- `~start_y`, `~end_y` (float, m) — Y bounds
- `~x`, `~z` (float, m) — fixed X/Z
- `~velocity` (float, m/s)
- `~update_rate` (float, Hz)

Run (example):
    
    rosrun <your_pkg> dynamic_obstacle.py \
      _model_name:=moving_box _type:=box \
      _start_y:=0.6 _end_y:=-0.6 _x:=0.5 _z:=0.3 \
      _velocity:=0.2 _update_rate:=50

Launch snippet:
    
    <node pkg="your_pkg" type="dynamic_obstacle.py" name="moving_obstacle_controller" output="screen">
      <param name="type" value="box"/>
      <param name="model_name" value="moving_box"/>
      <param name="start_y" value="0.6"/>
      <param name="end_y" value="-0.6"/>
      <param name="x" value="0.5"/>
      <param name="z" value="0.3"/>
      <param name="velocity" value="0.2"/>
      <param name="update_rate" value="50"/>
    </node>

---

## 3) kinematics_validation.py
**Purpose**
- Compare **FK** tool-frame pose/velocity and **Jacobian** predictions to **real** EE pose/velocity (from TF).  
- Log results to `kinematics_test_results.csv`.  
- Publish small test velocities to joints 1–3 (others zero) at 10 Hz.

**How it works**
- Uses `ControlPoints()` (your module) to get:
  - current joints `q` and joint velocities `qd`
  - tool-frame FK pose (position + quaternion)
  - Jacobian via `tool_frame_cp.evaluate_jacobian(q)`
- Gets **real** EE pose from TF: `base_link → kinova_arm_tool_frame` (position, quaternion → RPY).
- Estimates **real** EE velocity by linear fit over a sliding window (`window_size = 5`) of recent positions.
- Computes:
  - Position error: `‖p_fk − p_real‖`
  - Orientation error (RPY): `‖RPY_fk − RPY_real‖`
  - EE velocity error: `‖J*qd − v_real‖`
  - Joint velocity error: `‖qd − pinv(J)*v_real‖`
- Publishes test velocities: `[0.06, 0.02, 0.01, 0, 0, 0]` rad/s.

CSV columns:
    
    Time,
    Joint Velocities, Joint Velocities Computed, Joint Velocity Error,
    EE Position FK, EE Position Real, Position Error,
    EE Velocity FK, EE Velocity Real, Velocity Error,
    Orientation FK (RPY), Orientation Real (RPY), Orientation Error (rad)

Run:
    
    rosrun VVF kinematics_validation.py

---

## Quick troubleshooting
- **Arm not moving** → verify velocity controller topics exist and accept `Float64`; fix joint indices in callback.
- **Obstacle not moving** → ensure Gazebo is running, `model_name` exists, and `/gazebo/set_model_state` is available.
- **Validation TF warnings** → check TF publishers and frame names; avoid near-singular `J` (reduce test speeds or change pose).
- **No CSV** → check write permissions in the working directory.
