# ROS1 Noetic in Docker (GUI-ready)

This repo provides a clean **ROS Noetic (Ubuntu 20.04)** development environment with RViz/rqt/Gazebo support.
Your host workspace `ros_ws/` is mounted inside the container at **`/ros_ws`**.

The Dockerfile auto-sources:
- `/opt/ros/noetic/setup.bash`
- `/ros_ws/{devel|install}/setup.bash` (when they exist)

so every interactive shell “just has ROS”.

---

## Build the image
```bash
# from the repo root (contains Dockerfile/ and ros_ws/)
docker build --no-cache -f Dockerfile/Dockerfile -t kinova_noetic .
```

## run the image
```bash
xhost +local:root  # once per login session

docker run --rm -it --name kinova_noetic \
  --net=host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device=/dev/dri:/dev/dri \
  -v $(pwd)/ros_ws:/ros_ws \
  --user $(id -u):$(id -g) \
  kinova_noetic
```

## inside the container the first time:
```bash
# 1) Make sure git-lfs is installed in the container
sudo apt-get update
sudo apt-get install -y git-lfs

# 2) Pull LFS objects and submodules for ros_kortex
cd /ros_ws/src/ros_kortex
git lfs install
git lfs pull
git submodule update --init --recursive

# (Optional) sanity check: you should find .proto files
# and/or generated pb.h after a build
find . -name "ActuatorCyclic.proto" -o -name "ActuatorCyclic.pb.h"

# 3) Clean the partial build
cd /ros_ws
catkin clean -y kortex_driver kortex_examples kortex_gazebo

# 4) Make sure protoc is available (it should be from your Dockerfile)
protoc --version

catkin build
source devel/setup.bash

```
## Quick test 
```bash
echo $ROS_DISTRO              # should be: noetic
rospack find gazebo_ros
rospack find velodyne_pointcloud
rospack find sick_tim
roscore
```
## To open a second terminal on the host:
```bash
docker exec -it kinova_noetic bash
```
then for test (with roscore up and running):
```bash
rviz
```

## Remove any old container/image and clean cache (host)
```bash
# stop/remove container (ignore errors if it doesn't exist)
docker rm -f kinova_noetic 2>/dev/null || true

# remove the image tag
docker rmi -f kinova_noetic:latest 2>/dev/null || true

# optional cleanups (free space)
docker image prune -f
docker builder prune -f
```