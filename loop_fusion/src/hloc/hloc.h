#ifndef TEST_HLOC_H
#define TEST_HLOC_H

#include <torch/script.h>
#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "../utility/tic_toc.h"

#define SuperPointPath "/home/jesse/workspace/catkin_ws_ov/src/ov_hloc/support_files/models/SuperPoint_300.pt"
#define NetVLADPath "/home/jesse/workspace/catkin_ws_ov/src/ov_hloc/support_files/models/NetVLAD.pt"
#define SuperGluePath "/home/jesse/workspace/catkin_ws_ov/src/ov_hloc/support_files/models/SuperGlue_outdoor.pt"
#define UltraPointPath "/home/jesse/workspace/catkin_ws_ov/src/ov_hloc/support_files/models/UltraPoint.pt"

class SuperPoint {
public:
    static SuperPoint& self();
    static void Extract(
            const cv::Mat &image,
            std::vector<cv::Point2f> &kpts,
            std::vector<float> &scrs,
            cv::Mat &desc
            );
private:
    torch::jit::script::Module model;
    SuperPoint();
    void IExtract(
            const cv::Mat &image,
            std::vector<cv::Point2f> &kpts,
            std::vector<float> &scrs,
            cv::Mat &desc
            );
};

class NetVLAD {
public:
    static NetVLAD& self();
    static void Extract(
            const cv::Mat &image,
            cv::Mat &desc
    );
private:
    torch::jit::script::Module model;
    NetVLAD();
    void IExtract(
            const cv::Mat &image,
            cv::Mat &desc
    );
};

class SuperGlue {
public:
    static SuperGlue& self();
    static void Match(
            std::vector<cv::Point2f> &kpts0,
            std::vector<float> &scrs0,
            cv::Mat &desc0,
            int height0, int width0,
            std::vector<cv::Point2f> &kpts1,
            std::vector<float> &scrs1,
            cv::Mat &desc1,
            int height1, int width1,
            std::vector<int> &match_index,
            std::vector<float> &match_score
    );
private:
    torch::jit::script::Module model;
    SuperGlue();
    void IMatch(
            std::vector<cv::Point2f> &kpts0,
            std::vector<float> &scrs0,
            cv::Mat &desc0,
            int height0, int width0,
            std::vector<cv::Point2f> &kpts1,
            std::vector<float> &scrs1,
            cv::Mat &desc1,
            int height1, int width1,
            std::vector<int> &match_index,
            std::vector<float> &match_score
    );
};

class UltraPoint {
public:
    static UltraPoint& self();
    static void Extract(
            const cv::Mat &image,
            std::vector<cv::Point2f> &kpts,
            std::vector<float> &scrs,
            cv::Mat &desc
    );
private:
    torch::jit::script::Module model;
    UltraPoint();
    void IExtract(
            const cv::Mat &image,
            std::vector<cv::Point2f> &kpts,
            std::vector<float> &scrs,
            cv::Mat &desc
    );
};

#endif //TEST_HLOC_H