FROM ros:noetic
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      python3-pip \
      python3-catkin-tools \
      git \
      && rm -rf /var/lib/apt/lists/*

  # Copy the entire project
  COPY . /NICO-software/

  # Set working directory
  WORKDIR /NICO-software/api

  # Install exact Python dependencies that work
  RUN apt-get update && apt-get install -y \
     python3-pip \
     python3-catkin-tools \
     python3-rosdep \
     python3-rosinstall \
     python3-rosinstall-generator \
     python3-wstool \
     build-essential \
     git \
     vim \
     && rm -rf /var/lib/apt/lists/*

  # Build ROS workspace
  WORKDIR /NICO-software/api
  RUN rm -rf /NICO-software/api/build/CMakeCache.txt
  RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"


  # Setup environment
  RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
  RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
  RUN echo "export COPPELIASIM_ROOT=/home/hb/Downloads/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.bashrc

  WORKDIR /NICO-software
  CMD ["/bin/bash"]
