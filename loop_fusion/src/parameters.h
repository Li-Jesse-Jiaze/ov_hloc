#pragma once

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

extern camodocal::CameraPtr m_camera;
extern double max_focallength;
extern double LOOP_THRESHOLD;
extern double RELOC_THRESHOLD;
extern double PNP_INFLATION;
extern int RECALL_IGNORE_RECENT_COUNT;
extern double MAX_THETA_DIFF;
extern double MAX_POS_DIFF;
extern int MIN_LOOP_NUM;
extern Eigen::Vector3d tic;
extern Eigen::Matrix3d qic;
extern ros::Publisher pub_match_img;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string POSE_GRAPH_SAVE_PATH;
extern int ROW;
extern int COL;
extern std::string VINS_RESULT_PATH;
extern int DEBUG_IMAGE;
extern int LOAD_PREVIOUS_POSE_GRAPH;
