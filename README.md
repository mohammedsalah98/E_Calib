# NVSCircles_calib
Event-driven calibration of neuromorphic vision sensors

[![Novel Event-Driven Calibration of Bio-Inspired Neuromorphic Vision Sensors]

(https://www.dropbox.com/s/dliabxsd6qrl060/video_thumbnail.png?dl=0)](https://www.dropbox.com/s/m0jgabgkos940lj/NVSCircles_video.mp4?dl=0)

#
You can find the PDF of the paper [here]().
If you use this code in an academic context please cite this publication:

```bibtex
@Article{nvscircles,
  author        = {Mohammed Salah and Abdulla Ayyad and Muhammad Humais and Abdulqader Abusafieh and Yahya Zweiri},
  title         = {Novel Event-Driven Calibration of Bio-Inspired Neuromorphic Vision Sensors},
  journal       = {Arxiv},
  url           = {},
  year          = 2022
}
```

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
rosbag record /dvs/events
```

### Step 3: Convert the bag file to h5 file
Create NVSCircles conda environment:
```
git clone https://github.com/mohammedsalah98/NVSCircles_calib.git
cd NVSCircles_calib
conda env create -f environment.yml
conda activate NVSCircles
```
Convert the recorded bag to h5:
```
./bag_to_h5.py
```

The code is interactive and asks for the h5 data directory on the go. The h5 data will be saved in the dataset directory.

### Step 4: Run the calibration script
After converting the bag file to h5, run the calibration script:
```
./NVSCircles_calib.py
```

This code is also interactive and asks for the required data after running, including the resolution of the sensor and the pattern properties.

## Disclaimer
We will soon release the code also as a ros node to do calibration of neuromorphic vision sensors online.
