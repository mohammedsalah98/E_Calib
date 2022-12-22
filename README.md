# NVSCircles_calib
Event-driven calibration of neuromorphic vision sensors

## Supported platforms

Tested on the following platforms:

- ROS Noetic/Melodic
- Ubuntu 18.04 and 20.04 LTS

Tested on the following hardware:

- DAVIS346
- DVXplorer

## Prerequisites
The following package needs to be in your catkin workspace to record event data sequences:

[rpg_dvs_ros](https://github.com/uzh-rpg/rpg_dvs_ros)

An asymmetric circles grid or you can use our dataset on: [NVS-ACircles](https://www.dropbox.com/sh/jxxsscijfeby3px/AADw3GzuV08WAo3q2WeBBonoa?dl=0)

## Running the software
### Step 1: Create your catkin workspace
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/uzh-rpg/rpg_dvs_ros.git
git clone https://github.com/catkin/catkin_simple.git
cd .. && catkin build
```

### Step 2: Record your event data sequence
```
cd catkin_ws
source devel/setup.bash
roslaunch dvs_renderer davis_color.launch (FOR DAVIS346)
roslaunch dvs_renderer dvxplorer_mono.launch (FOR DVXplorer)
rosbag record /dvs/events
```
