#include "hloc.h"

SuperPoint &SuperPoint::self() {
    static SuperPoint instance;
    return instance;
}

SuperPoint::SuperPoint() {
    model = torch::jit::load(SuperPointPath);
    ROS_INFO("Loaded SuperPoint");
}

void SuperPoint::Extract(const cv::Mat &image, std::vector<cv::Point2f> &kpts, std::vector<float> &scrs, cv::Mat &desc) {
    // TicToc tt;
    self().IExtract(image, kpts, scrs, desc);
    // printf("SuperPoint %fms\n", tt.toc());
}

void SuperPoint::IExtract(const cv::Mat &image, std::vector<cv::Point2f> &kpts, std::vector<float> &scrs, cv::Mat &desc) {
    kpts.clear();
    scrs.clear();
    cv::Mat image_gray = image.clone();
    // resize image if size > max size (1024)
    // record the original size
    float scale = 1;
    int img_width_ori = image_gray.cols, img_width_new = image_gray.cols;
    int img_height_ori = image_gray.rows, img_height_new = image_gray.rows;
    if (std::max(img_width_ori, img_height_ori) > 1024) {
        scale = 1024.f / std::max(img_width_ori, img_height_ori);
        img_width_new = img_width_ori * scale;
        img_height_new = img_height_ori * scale;
        cv::resize(image_gray, image_gray, cv::Size(img_width_new, img_height_new), 0, 0, cv::INTER_LINEAR);
    }
    // convert cv::Mat to torch::Tensor [batch x C x H x W]
    image_gray = cv::dnn::blobFromImage
            (
                    image_gray, 1.f / 255.f, // scale factor
                    cv::Size(), // spatial size for output image
                    cv::Scalar(), // mean
                    true, // swapRB: BGR to RGB
                    false, // crop
                    CV_32F // Depth of output blob. Choose CV_32F or CV_8U.
            );
    torch::Tensor img_tensor = torch::from_blob(image_gray.data, {1, 1, img_height_new, img_width_new}).to(
            torch::kCUDA);

    // put image into model
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> torch_inputs;
    torch::jit::IValue torch_outputs;
    torch_inputs.emplace_back(img_tensor);
    torch_outputs = model.forward(torch_inputs);
    auto outputs_tuple = torch_outputs.toTuple();
    auto keypoints = outputs_tuple->elements()[0].toTensorVector()[0].to(torch::kCPU);   // N x 2
    auto scores = outputs_tuple->elements()[1].toTensorVector()[0].to(torch::kCPU);
    auto descriptors = outputs_tuple->elements()[2].toTensorVector()[0].to(torch::kCPU); // 256 x N

    // reduce position of keypoints
    keypoints = (keypoints / scale).round();
    // convert torch::Tensor to std::vector<cv::KeyPoint>
    for (int i = 0; i < keypoints.size(0); i++) {
        kpts.emplace_back(keypoints[i][0].item<float>(), keypoints[i][1].item<float>());
    }
    // convert torch::Tensor to std::vector<std::float>
    for (int i = 0; i < scores.size(0); i++) {
        scrs.push_back(scores[i].item<float>());
    }
    // convert torch::Tensor to cv::Mat
    cv::Mat mat_desc(descriptors.size(0), descriptors.size(1), CV_32FC1);
    std::memcpy(mat_desc.data, descriptors.data_ptr(), sizeof(float) * descriptors.numel());
    desc = mat_desc;
}


NetVLAD &NetVLAD::self() {
    static NetVLAD instance;
    return instance;
}

NetVLAD::NetVLAD() {
    model = torch::jit::load(NetVLADPath);
    ROS_INFO("Loaded NetVLAD");
}

void NetVLAD::Extract(const cv::Mat &image, cv::Mat &desc) {
    // TicToc tt;
    self().IExtract(image, desc);
    // printf("NetVLAD %fms\n", tt.toc());
}

void NetVLAD::IExtract(const cv::Mat &image, cv::Mat &desc) {
    cv::Mat _image = image.clone();
    cv::cvtColor(_image, _image, cv::COLOR_GRAY2BGR);
    // resize image if size > max size (1024)
    // record the original size
    float scale = 1;
    int img_width_ori = _image.cols, img_width_new = _image.cols;
    int img_height_ori = _image.rows, img_height_new = _image.rows;
    if (std::max(img_width_ori, img_height_ori) > 1024) {
        scale = 1024.f / std::max(img_width_ori, img_height_ori);
        img_width_new = img_width_ori * scale;
        img_height_new = img_height_ori * scale;
        cv::resize(_image, _image, cv::Size(img_width_new, img_height_new), 0, 0, cv::INTER_LINEAR);
    }
    // convert cv::Mat to torch::Tensor [batch x C x H x W]
    _image = cv::dnn::blobFromImage
            (
                    _image, 1.f / 255.f, // scale factor
                    cv::Size(), // spatial size for output image
                    cv::Scalar(), // mean
                    true, // swapRB: BGR to RGB
                    false, // crop
                    CV_32F // Depth of output blob. Choose CV_32F or CV_8U.
            );
    torch::Tensor img_tensor = torch::from_blob(_image.data, {1, 3, img_height_new, img_width_new}).to(
            torch::kCUDA);

    // put image into model
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> torch_inputs;
    torch::jit::IValue torch_outputs;
    torch_inputs.emplace_back(img_tensor);
    torch_outputs = model.forward(torch_inputs);
    auto descriptors = torch_outputs.toTensor().to(torch::kCPU);

    // convert torch::Tensor to cv::Mat
    cv::Mat mat_desc(descriptors.size(0), descriptors.size(1), CV_32FC1);
    std::memcpy(mat_desc.data, descriptors.data_ptr(), sizeof(float) * descriptors.numel());
    desc = mat_desc;
}


SuperGlue &SuperGlue::self() {
    static SuperGlue instance;
    return instance;
}

SuperGlue::SuperGlue() {
    model = torch::jit::load(SuperGluePath);
    ROS_INFO("Loaded SuperGlue");
}

void SuperGlue::Match(std::vector<cv::Point2f> &kpts0, std::vector<float> &scrs0, cv::Mat &desc0, int height0, int width0,
                 std::vector<cv::Point2f> &kpts1, std::vector<float> &scrs1, cv::Mat &desc1, int height1, int width1,
                 std::vector<int> &match_index, std::vector<float> &match_score) {
    // TicToc tt;
    self().IMatch(kpts0, scrs0, desc0, height0, width0,
                  kpts1, scrs1, desc1, height1, width1,
                  match_index, match_score);
    // printf("SuperGlue %fms\n", tt.toc());
}

void SuperGlue::IMatch(std::vector<cv::Point2f> &kpts0, std::vector<float> &scrs0, cv::Mat &desc0, int height0, int width0,
                  std::vector<cv::Point2f> &kpts1, std::vector<float> &scrs1, cv::Mat &desc1, int height1, int width1,
                  std::vector<int> &match_index, std::vector<float> &match_score) {
    match_index.clear();
    match_score.clear();
    auto k0 = torch::from_blob(kpts0.data(), {1, int(kpts0.size()), 2}).to(torch::kCUDA);
    auto k1 = torch::from_blob(kpts1.data(), {1, int(kpts1.size()), 2}).to(torch::kCUDA);
    auto s0 = torch::from_blob(scrs0.data(), {1, int(scrs0.size())}).to(torch::kCUDA);
    auto s1 = torch::from_blob(scrs1.data(), {1, int(scrs1.size())}).to(torch::kCUDA);
    auto d0 = torch::from_blob(desc0.clone().data, {1, desc0.rows, desc0.cols}).to(torch::kCUDA);
    auto d1 = torch::from_blob(desc1.clone().data, {1, desc1.rows, desc1.cols}).to(torch::kCUDA);
    auto size0 = torch::tensor({height0, width0}).to(torch::kCUDA);
    auto size1 = torch::tensor({height1, width1}).to(torch::kCUDA);

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> torch_inputs;
    torch::jit::IValue torch_outputs;

    torch_inputs.emplace_back(k0);
    torch_inputs.emplace_back(s0);
    torch_inputs.emplace_back(d0);
    torch_inputs.emplace_back(size0);
    torch_inputs.emplace_back(k1);
    torch_inputs.emplace_back(s1);
    torch_inputs.emplace_back(d1);
    torch_inputs.emplace_back(size1);

    torch_outputs = model.forward(torch_inputs);
    auto outputs_tuple = torch_outputs.toTuple();
    auto index = outputs_tuple->elements()[0].toTensor();
    auto score = outputs_tuple->elements()[2].toTensor();
    for (int i = 0; i < index.sizes()[1]; i++) {
        match_index.push_back(index[0][i].item<int>());
        match_score.push_back(score[0][i].item<float>());
    }
}

UltraPoint &UltraPoint::self() {
    static UltraPoint instance;
    return instance;
}

UltraPoint::UltraPoint() {
    model = torch::jit::load(UltraPointPath);
    ROS_INFO("Loaded UltraPoint");
}

void UltraPoint::Extract(const cv::Mat &image, std::vector<cv::Point2f> &kpts, std::vector<float> &scrs, cv::Mat &desc) {
    // TicToc tt;
    self().IExtract(image, kpts, scrs, desc);
    // printf("UltraPoint %fms\n", tt.toc());
}

void UltraPoint::IExtract(const cv::Mat &image, std::vector<cv::Point2f> &kpts, std::vector<float> &scrs,
                          cv::Mat &desc) {
    scrs.clear();
    cv::Mat image_gray = image.clone();
    auto keypoints_tensor = torch::from_blob(kpts.data(), {1, int(kpts.size()), 2}).to(torch::kCUDA);
    // resize image if size > max size (1024)
    // record the original size
    float scale = 1;
    int img_width_ori = image_gray.cols, img_width_new = image_gray.cols;
    int img_height_ori = image_gray.rows, img_height_new = image_gray.rows;
    if (std::max(img_width_ori, img_height_ori) > 1024) {
        scale = 1024.f / std::max(img_width_ori, img_height_ori);
        img_width_new = img_width_ori * scale;
        img_height_new = img_height_ori * scale;
        cv::resize(image_gray, image_gray, cv::Size(img_width_new, img_height_new), 0, 0, cv::INTER_LINEAR);
        keypoints_tensor = (keypoints_tensor * scale).round();
    }
    // convert cv::Mat to torch::Tensor [batch x C x H x W]
    image_gray = cv::dnn::blobFromImage
            (
                    image_gray, 1.f / 255.f, // scale factor
                    cv::Size(), // spatial size for output image
                    cv::Scalar(), // mean
                    true, // swapRB: BGR to RGB
                    false, // crop
                    CV_32F // Depth of output blob. Choose CV_32F or CV_8U.
            );
    torch::Tensor img_tensor = torch::from_blob(image_gray.data, {1, 1, img_height_new, img_width_new}).to(
            torch::kCUDA);

    // put image into model
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> torch_inputs;
    torch::jit::IValue torch_outputs;
    torch_inputs.emplace_back(img_tensor);
    torch_inputs.emplace_back(keypoints_tensor.to(torch::kLong));
    torch_outputs = model.forward(torch_inputs);
    auto outputs_tuple = torch_outputs.toTuple();
    auto scores = outputs_tuple->elements()[0].toTensor().to(torch::kCPU);
    auto descriptors = outputs_tuple->elements()[1].toTensorVector()[0].to(torch::kCPU); // 256 x N

    // convert torch::Tensor to std::vector<std::float>
    for (int i = 0; i < scores.size(0); i++) {
        scrs.push_back(scores[i].item<float>());
    }
    // convert torch::Tensor to cv::Mat
    cv::Mat mat_desc(descriptors.size(0), descriptors.size(1), CV_32FC1);
    std::memcpy(mat_desc.data, descriptors.data_ptr(), sizeof(float) * descriptors.numel());
    desc = mat_desc;
}
