The location procedure is based on 'scene' and 'unknow' repositories.

We first drive robot to establish the surroundings. During the mapping process,

 recognition is carried out to generate semantic relationships(xy.txt). 
 
Then the robot randomly locates at a position on the map, conducts an environmental scan of the location and performs matching.

   1.Execute mapping
=======
If you have any other mapping method, it's ok
RTAB-Map's ROS package.

For more information, demos and tutorials about this package, visit [rtabmap_ros](http://wiki.ros.org/rtabmap_ros) page on ROS wiki.

For the RTAB-Map libraries and standalone application, visit [RTAB-Map's home page](http://introlab.github.io/rtabmap) or [RTAB-Map's wiki](https://github.com/rtabmap).



RTAB-Map is released as binaries in the ROS distribution.

```bash
sudo apt install ros-$ROS_DISTRO-rtabmap-ros
```

When launching `rtabmap_ros`'s nodes, if you have the error `error while loading shared libraries...`, try `ldconfig` or add the next line at the end of your `~/.bashrc` to fix it:
    
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/noetic/lib/x86_64-linux-gnu
```




Quick Start
This section shows how to install RTAB-Map ros-pkg on **ROS Melodic/Noetic** (Catkin build).

* The next instructions assume that you have set up your ROS workspace using this [tutorial](http://wiki.ros.org/catkin/Tutorials/create_a_workspace). The workspace path is `~/catkin_ws` and your `~/.bashrc` contains:
 
    ```bash
    $ source /opt/ros/$ROS_DISTRO/setup.bash
    $ source ~/catkin_ws/devel/setup.bash
    ```

 0. Required dependencies
     * The easiest way to get all them (Qt, PCL, VTK, OpenCV, ...) is to install/uninstall rtabmap binaries:
          ```bash
          sudo apt install ros-$ROS_DISTRO-rtabmap ros-$ROS_DISTRO-rtabmap-ros
          sudo apt remove ros-$ROS_DISTRO-rtabmap ros-$ROS_DISTRO-rtabmap-ros
          ```
 
 1. Optional dependencies
     * If you want SURF/SIFT on Melodic/Noetic, you have to build [OpenCV]([OpenCV](http://opencv.org/)) from source to have access to *xfeatures2d* and *nonfree* modules (note that SIFT is not in *nonfree* anymore since OpenCV 4.4.0). Install it in `/usr/local` (default) and rtabmap library should link with it instead of the one installed in ROS. 
         * On Melodic/Noetic, build from source with *xfeatures2d* module (and *nonfree* module if needed) the same OpenCV version already installed on the system. You will then avoid breaking `cv_bridge` with `rtabmap_ros`. If you want to install a more recent OpenCV version, I recommend to uninstall `libopencv*` libraries (with all ros packages depending on it) and rebuild all those ros packages in your catkin workspace (to make sure `cv_bridge` is linked on the OpenCV version you just compiled).
  
    * [g2o](https://github.com/RainerKuemmerle/g2o): Should be already installed by `ros-$ROS_DISTRO-libg2o`.

    * [GTSAM](https://gtsam.org/get_started/): Install via PPA to avoid building from source. If you install from source, make sure to build with `cmake  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON`.
    
    
2. Install RTAB-Map standalone libraries. **Do not clone in your Catkin workspace**.
    ```bash
    cd ~
    git clone https://github.com/rtabmap.git rtabmap
    cd rtabmap/build
    cmake ..  [<---double dots included]
    make -j6
    sudo make install
    ```

3. Install RTAB-Map ros-pkg in your src folder of your Catkin workspace.
 
    ```bash
    cd ~/catkin_ws
    git clone https://github.com/rtabmap_ros.git src/rtabmap_ros
    catkin_make -j4
    ```
    * Use `catkin_make -j1` if compilation requires more RAM than you have (e.g., some files require up to ~2 GB to build depending on gcc version).
    * Options:
        * Add `-DRTABMAP_SYNC_MULTI_RGBD=ON` to `catkin_make` if you plan to use multiple cameras.
        * Add `-DRTABMAP_SYNC_USER_DATA=ON` to `catkin_make` if you plan to use user data synchronized topics.

2、Execute 3D detections
=======
1. Quick start

   source devel/setup.bash
   roslaunch gsm_node scenenn_dataset.launch bag_file:=/home/3ddetections/src/test.bag 
   catkin build mask_rcnn_ros depth_segmentation gsm_node global_segment_map
  

3、Execute  Clustering Triangle
=======
Quick start
*Generating the position data xy.txt from 3d detections
```
python train.py
python  TransformerEncoderLayer.py
python shuzu.py
python word2vectors.py
```
*Visualizing the triangle
```
Execute visualizer.cc

python bound7.py
python connect6.py

```
4、Execute  matching
=======
### Run the localizer
Once you get your pcd map and configuration ready, run the localizer with:

Execute map_loader.cpp

```bash
# open a roscore
roscore
# in other terminal
cd catkin_ws
source devel/setup.bash
# use rosbag sim time if you are playing a rosbag!!!
rosparam set use_sim_time true
# launch the ndt_localizer node
roslaunch ndt_localizer ndt_localizer.launch
```
