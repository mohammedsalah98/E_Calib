# E-Calib
## A Fast, Robust and Accurate Calibration Tool for Event Cameras 

[![E-Calib:](https://github.com/mohammedsalah98/E_Calib/blob/master/video_thumbnail.png)](https://youtu.be/4giQn6rt-48)

#
You can find the PDF of the paper [here]().
If you use this code in an academic context please cite this publication:

```bibtex
@ARTICLE{E-Calib,
  author={Salah, Mohammed and Ayyad, Abdulla and Humais, Muhammad and Gehrig, Daniel and Abusafieh, Abdelqader and Seneviratne, Lakmal and Scaramuzza, Davide and Zweiri, Yahya},
  journal={IEEE Transactions on Image Processing}, 
  title={E-Calib: A Fast, Robust, and Accurate Calibration Toolbox for Event Cameras}, 
  year={2024},
  volume={33},
  pages={3977-3990},
  doi={10.1109/TIP.2024.3410673}}
```

## Code Structure and ECam_ACircles Dataset Outline:
![Alt text](https://github.com/mohammedsalah98/E_Calib/blob/master/ECam_ACircles.png)

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

An asymmetric circles grid or you can use our dataset on: [ECam-ACircles](https://drive.google.com/drive/folders/1m_l6S-mTReuIfsrB6UnwOuPu111-0Mfz?usp=sharing)

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
Create ECalib conda environment:
```
git clone https://github.com/mohammedsalah98/E_Calib.git
cd E_Calib
conda env create -f environment.yml
conda activate ECalib
```
Convert the recorded bag to h5:
```
./bag_to_h5.py
```

The code is interactive and asks for the h5 data directory on the go. The h5 data will be saved in the dataset directory.

### Step 4: Run the calibration script
After converting the bag file to h5, run the calibration script:
```
./E_Calib.py
```

This code is also interactive and asks for the required data after running, including the resolution of the sensor and the calibration pattern properties.

If you are looking for improved calibration accuracy, we rather recommend running the MATLAB script 'matlab/calib_img_pts_lsqnonlin.m'. The code follows the same structure of the aforementioned python script.

## Disclaimer
We will soon release the code also as a ros node to do calibration of event cameras online.
