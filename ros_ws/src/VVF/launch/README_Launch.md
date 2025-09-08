# Launch Files — `bringup.launch` & `dynamic_obstacle.launch`

Everything needed to start Gazebo with the Jackal + Kinova arm and to spawn a moving obstacle.

---

## `bringup.launch`

**Purpose**
- Start Gazebo with a selectable world.
- Build `robot_description` (Jackal + Kinova URDF via xacro).
- Spawn the robot.
- Load and spawn **6 velocity controllers** for the arm.
- Run `robot_state_publisher`.

**Key args**
- `use_sim_time` (default `true`) — use `/clock` from Gazebo  
- `gui` / `headless` — show/hide Gazebo UI  
- `front_laser` (default `false`) → chooses Jackal config (`front_laser` vs `base`)  
- `config` — resolved from `front_laser`  
- `prefix` (default `kinova_arm_`) — controller name prefix  
- `cyclic_data_publish_rate` (default `40`)  
- `world` (default `empty`) — `empty|one_cylinder|two_obstacles|two_spheres|/abs/path.world`  
- `world_file` — path resolved from `world` (via `$(eval ...)`)  

**What it launches**
- Gazebo (`gazebo_ros/empty_world.launch`) with `world_name=$(arg world_file)`  
- Builds `robot_description` using:
  - `jackal_description/scripts/env_run`
  - `jackal_description/urdf/configs/$(arg config)`
  - `VVF/urdf/jackal.urdf.xacro`
- Loads arm controllers from `VVF/config/joint_velocity_controllers.yaml`
- Spawns URDF as model `jackal` at `(x=0, y=0, z=1)`
- Spawns controller manager for:
  - `kinova_arm_joint_1_velocity_controller`
  - `kinova_arm_joint_2_velocity_controller`
  - `kinova_arm_joint_3_velocity_controller`
  - `kinova_arm_joint_4_velocity_controller`
  - `kinova_arm_joint_5_velocity_controller`
  - `kinova_arm_joint_6_velocity_controller`
- Runs `robot_state_publisher`

**Run examples**
    
    # Default (empty world, GUI)
    roslaunch VVF bringup.launch

    # Choose a packaged world
    roslaunch VVF bringup.launch world:=two_obstacles

    # Headless
    roslaunch VVF bringup.launch gui:=false headless:=true

    # Jackal with front laser config
    roslaunch VVF bringup.launch front_laser:=true

    # Custom world file
    roslaunch VVF bringup.launch world:=/home/user/worlds/my_scene.world

**Notes**
- Requires: `gazebo_ros`, `xacro`, `jackal_description`, `jackal_control`, `controller_manager`, `robot_state_publisher`, and your `VVF` package.
- Make sure controller names in `joint_velocity_controllers.yaml` match the topics you use.

---

## `dynamic_obstacle.launch`

**Purpose**
- Spawn an SDF obstacle (hand / sphere / cylinder) and **oscillate it along Y** using `dynamic_obstacle.py`.

**Args**
- `type` (default `hand`) — `hand|sphere|cylinder` → selects SDF
- `model_name` (default `moving_object`)
- `x`, `z` (defaults `0.6`, `0.6`) — fixed X/Z (m)
- `start_y`, `end_y` (defaults `2.0`, `-2.0`) — Y-bounds (m)
- `velocity` (default `0.22`) — m/s
- `update_rate` (default `50`) — Hz  
- Internally resolves `sdf_path` from:
  - `VVF/models/hand_object/hand_object.sdf`
  - `VVF/models/moving_sphere.sdf`
  - `VVF/models/moving_cylinder.sdf`

**What it launches**
- Spawns the SDF model at `(x, start_y, z)` named `model_name`  
- Starts `VVF/dynamic_obstacle.py` with the above parameters (moves the model back-and-forth along Y)

**Run examples**
    
    # Default: hand object, moderate speed
    roslaunch VVF dynamic_obstacle.launch

    # Faster cylinder
    roslaunch VVF dynamic_obstacle.launch type:=cylinder velocity:=0.4

    # Sphere at different position/bounds
    roslaunch VVF dynamic_obstacle.launch type:=sphere x:=0.8 z:=0.5 start_y:=1.0 end_y:=-1.0

**Typical sequence**
    
    roslaunch VVF bringup.launch world:=one_cylinder
    roslaunch VVF dynamic_obstacle.launch type:=cylinder velocity:=0.25

**Troubleshooting**
- Check Gazebo is running and `/gazebo/set_model_state` exists:
    
        rosservice list | grep set_model_state

- If the obstacle spawns but doesn’t move:
  - Ensure `model_name` matches the spawned name
  - Confirm `dynamic_obstacle.py` is executable
  - Adjust `start_y/end_y` and `velocity` relative to `update_rate`
