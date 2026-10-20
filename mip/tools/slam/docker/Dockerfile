# The ORB-SLAM3 image the SLAM interfaces run (`mip/orbslam3:latest`, the name
# mip/tools/slam/world/_slam.py starts). It holds only the SLAM library and its Python
# binding: the server script, world/_slam_server.py, comes in through a read-only mount
# at run time, so a code change never needs a rebuild.
#
#   docker build -t mip/orbslam3:latest mip/tools/slam/docker            # ~12 min
#   docker build --build-arg JOBS=8 -t mip/orbslam3:latest mip/tools/slam/docker
#
# JOBS is the compiler parallelism; lower it on a machine short of RAM.
# The three sources are pinned to the commits of the image the paper's runs used;
# OpenCV 4.5.4 and numpy 1.21 come from Ubuntu 22.04's packages.

FROM ubuntu:22.04

ARG JOBS=4
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates cmake build-essential pkg-config \
    libboost-dev libboost-serialization-dev libboost-python-dev \
    libssl-dev libeigen3-dev \
    libgl1-mesa-dev libglew-dev libepoxy-dev \
    libopencv-dev \
    python3 python3-dev python3-numpy \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Pangolin v0.9.1 (7364b59), ORB-SLAM3's viewer dependency; the server never opens a window.
RUN git clone --recursive --depth 1 --branch v0.9.1 \
        https://github.com/stevenlovegrove/Pangolin /opt/src/Pangolin && \
    cmake -B /opt/src/Pangolin/build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF /opt/src/Pangolin && \
    cmake --build /opt/src/Pangolin/build -j${JOBS} && \
    cmake --install /opt/src/Pangolin/build && ldconfig

# ORB-SLAM3, Hello Robot's fork (it carries the Ubuntu 22.04 build fixes). Its build.sh
# hard-codes make -j8; JOBS replaces that.
RUN git clone https://github.com/hello-binit/ORB_SLAM3 /opt/src/ORB_SLAM3 && \
    cd /opt/src/ORB_SLAM3 && git checkout 4071a3eb98b95de915017bf236b0bbab3eb658a3 && \
    sed -i "s/make -j8/make -j${JOBS}/" build.sh && ./build.sh && \
    cd build && make install && \
    cd ../Thirdparty/Sophus/build && make install && ldconfig

# The boost-python binding: installs the `orbslam3` module the server imports.
RUN git clone https://github.com/hello-binit/ORB_SLAM3-PythonBindings \
        /opt/src/ORB_SLAM3-PythonBindings && \
    git -C /opt/src/ORB_SLAM3-PythonBindings checkout 20f9573ae1492eca36598a5057a3da8a5384627c && \
    cmake -B /opt/src/ORB_SLAM3-PythonBindings/build \
        -DCMAKE_BUILD_TYPE=Release /opt/src/ORB_SLAM3-PythonBindings && \
    cmake --build /opt/src/ORB_SLAM3-PythonBindings/build -j${JOBS} && \
    cmake --install /opt/src/ORB_SLAM3-PythonBindings/build && ldconfig

# The ORB vocabulary, extracted where the server loads it (/opt/orbslam3/ORBvoc.txt).
RUN mkdir -p /opt/orbslam3 && \
    tar -xzf /opt/src/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz -C /opt/orbslam3

ENV LD_LIBRARY_PATH=/usr/local/lib
RUN python3 -c "import orbslam3; print('orbslam3 binding OK')"
