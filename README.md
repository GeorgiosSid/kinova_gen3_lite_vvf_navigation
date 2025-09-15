# MASTER README — Quick Start (BringUp (Ros, Gazebo the Jackal + Kinova) and Run a Simulation of the D-APF) 

**Read and follow** `Dockerfile/Dockerfile_README.md` to set up the environment (Docker, deps, `/ros_ws`).

**Always start `roscore` before launching or running anything.**

```bash
roscore
```

---

## Gazebo + Robot

```bash
# Default (empty world, GUI)
roslaunch VVF bringup.launch

# Choose a world (example)
roslaunch VVF bringup.launch world:=two_obstacles
```

More options in `ros_ws/src/VVF/launch/README_Launch.md`.

---

## Dynamic Obstacle Avoidance

```bash
rosrun VVF Dynamic_Obstacles_Avoidance.py
```

---

## Spawn a Moving Obstacle (example)

```bash
roslaunch VVF dynamic_obstacle.launch type:=cylinder velocity:=0.25
```

More details in `ros_ws/src/VVF/launch/README_Launch.md`.

