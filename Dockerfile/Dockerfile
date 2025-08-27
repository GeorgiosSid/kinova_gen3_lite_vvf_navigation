# Ubuntu 20.04 (Focal) + ROS Noetic
FROM ubuntu:20.04

SHELL ["/bin/bash","-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# ---- basics
RUN apt-get update && apt-get install -y \
    locales curl gnupg2 lsb-release sudo git nano bash-completion \
 && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---- ROS 1 repo + key (Focal)
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros1.list

# ---- core ROS + dev tools
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep python3-catkin-tools python3-vcstool build-essential \
 && rm -rf /var/lib/apt/lists/*

# ---- rosdep init
RUN rosdep init && rosdep update

# ---- extras (Gazebo 11, MoveIt, Jackal, Velodyne, etc.)
RUN apt-get update && apt-get install -y \
    gazebo11 \
    ros-noetic-gazebo-ros ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-plugins \
    ros-noetic-ros-control ros-noetic-ros-controllers \
    ros-noetic-joint-state-publisher-gui ros-noetic-xacro \
    ros-noetic-pcl-ros ros-noetic-pcl-conversions \
    ros-noetic-sick-tim \
    ros-noetic-hector-gazebo-plugins ros-noetic-stage-ros \
    # MoveIt (for completeness)
    ros-noetic-moveit ros-noetic-moveit-ros-planning-interface \
    ros-noetic-moveit-msgs ros-noetic-moveit-commander \
    ros-noetic-moveit-simple-controller-manager ros-noetic-moveit-setup-assistant \
    # Navigation bits & helpers
    ros-noetic-move-base ros-noetic-twist-mux ros-noetic-rosdoc-lite \
    ros-noetic-pointgrey-camera-description \
    # Jackal from apt (so they live under /opt/ros/noetic)
    ros-noetic-jackal-description \
    ros-noetic-jackal-msgs \
    ros-noetic-jackal-viz \
    ros-noetic-jackal-control \
    ros-noetic-jackal-navigation \
    # Sensors and localization used by Jackal stacks
    ros-noetic-lms1xx ros-noetic-map-server ros-noetic-teleop-twist-joy \
    ros-noetic-flir-camera-description ros-noetic-robot-localization ros-noetic-gmapping \
    # Velodyne stack
    ros-noetic-velodyne-description ros-noetic-velodyne-driver \
    ros-noetic-velodyne-gazebo-plugins ros-noetic-velodyne-laserscan \
    ros-noetic-velodyne-msgs ros-noetic-velodyne-pointcloud ros-noetic-velodyne-simulator \
    # Protobuf toolchain (needed by ros_kortex to generate *.pb.h)
    protobuf-compiler libprotobuf-dev libprotoc-dev python3-protobuf \
 && rm -rf /var/lib/apt/lists/*

# ---- optional Python bits + Conan 1.x
RUN apt-get update && apt-get install -y python3-pip python3-numpy \
 && pip3 install --no-cache-dir numpy-stl "conan==1.59" \
 && rm -rf /var/lib/apt/lists/*

# Ensure conan is always on PATH
RUN if [ -x /usr/local/bin/conan ]; then ln -sf /usr/local/bin/conan /usr/bin/conan; fi

# ---- non-root user
ARG USERNAME=noetic_user
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd -m -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

# ---- workspace
ENV WS=/ros_ws
RUN mkdir -p $WS/src
WORKDIR $WS

# ---- auto-source ROS + workspace
RUN printf '\n# ---- ROS Noetic auto source ----\n' >> /etc/bash.bashrc \
 && printf 'source /opt/ros/noetic/setup.bash\n' >> /etc/bash.bashrc \
 && printf '[ -f /ros_ws/devel/setup.bash ] && source /ros_ws/devel/setup.bash\n' >> /etc/bash.bashrc \
 && printf '[ -f /ros_ws/install/setup.bash ] && source /ros_ws/install/setup.bash\n' >> /etc/bash.bashrc

USER ${USERNAME}

# ---- preconfigure Conan
RUN conan --version \
 && conan config set general.revisions_enabled=1 \
 && (conan profile show default >/dev/null 2>&1 || conan profile new default --detect) \
 && conan profile update settings.compiler.libcxx=libstdc++11 default

CMD ["bash"]
