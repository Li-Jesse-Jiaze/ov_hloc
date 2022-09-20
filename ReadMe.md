
# OV_HLOC

Using **[Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)** instead of DBoW2 for loop closure. This was originally part of my undergraduate final project, where I worked on improving the loop closure module of VINS-Fusion. I found it just right for providing a loosely coupled pose graph for **[OpenVINS](https://github.com/rpng/open_vins)**.

![framework](image/cvci.svg)

Thanks to the excellent global pose graph optimization provided by VINS-Fusion, this project performs well on the EuRoC dataset.

![MH_05](image/MH_05.jpg)

In the application scenario you can use COLMAP to build SfM maps (using SuperPoint and NetVLAD). The use of a priori maps can give you more accurate positioning results and no accumulative errors.

## Dependencies

* Ubuntu and ROS - [noetic/Installation/Ubuntu - ROS Wiki](http://wiki.ros.org/noetic/Installation/Ubuntu)
  This will help you to install a series of dependencies such as OpenCV.

* OpenVINS - <https://docs.openvins.com/gs-installing.html>

* Ceres Solver - <https://github.com/ceres-solver/ceres-solver>

* PyTorch and libtorch - <https://pytorch.org/get-started/locally/>

  For libtorch, all you need to do is unzip it and fill the file path into [loop_hloc/CMakeLists.txt](loop_hloc/CMakeLists.txt) line 22.

  ```cmake
  # set your own libtorch path
  set(TORCH_PATH */libtorch/share/cmake/Torch)
  ```

## Installation Commands

```shell
# setup our workspace
mkdir -p ~/workspace/catkin_ws_ov/src/
cd ~/workspace/catkin_ws_ov/src/
# repositories to clone
git clone https://github.com/rpng/open_vins.git
git clone https://github.com/Li-Jesse-Jiaze/ov_hloc.git
# go back to root and build
cd ~/workspace/catkin_ws_ov
catkin build
```

## Download and Convert HF-Net

Download [SuperGluePretrainedNetwork/models/weights](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights) and place them in `support_files/Networks/weights`.

```shell
cd ~/workspace/catkin_ws_ov/src/ov_hloc/support_files/
python convert_model.py
```

## Run Example

``` shell
roslaunch ov_msckf subscribe.launch config:=euroc_mav # term 1
rosrun loop_hloc loop_hloc_node ~/workspace/catkin_ws_ov/src/ov_secondary/config/master_config.yaml # term 2
rviz # term 3
rosbag play your/dataset/path/V1_01_easy.bag # term 4
```

select `config/vins_rviz_config.rviz` as config in rviz

The pose graph was not very smooth when I tested it on my laptop(RTX 2060 Max-Q 65W). This is mainly because my NetVLAD is using a VGG16 (it tooks more than 50ms for each frame 😠). It would be better to use a lighter backbone (e.g. MobileNetV3) and fine-tuning NetVLAD for your application scenario.

Here's a simple [video](https://www.bilibili.com/video/bv1KP4y1F73M) of it working with VINS-Fusion.
