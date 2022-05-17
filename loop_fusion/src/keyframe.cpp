#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
                   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv,
                   vector<cv::Point2f> &_point_2d_normal, vector<double> &_point_id, int _sequence) {

    time_stamp = _time_stamp;
    index = _index;
    vio_T_w_i = _vio_T_w_i;
    vio_R_w_i = _vio_R_w_i;
    T_w_i = vio_T_w_i;
    R_w_i = vio_R_w_i;
    origin_vio_T = vio_T_w_i;
    origin_vio_R = vio_R_w_i;
    image = _image.clone();
    height = image.rows;
    width = image.cols;
    cv::resize(image, thumbnail, cv::Size(80, 60));
    point_3d = _point_3d;
    point_2d_uv = _point_2d_uv;
    point_2d_norm = _point_2d_normal;
    point_id = _point_id;
    has_loop = false;
    loop_index = -1;
    has_fast_point = false;
    loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
    sequence = _sequence;
    computeWindow();
    computeNew();
    if(!DEBUG_IMAGE)
        image.release();
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i,
                   Matrix3d &_R_w_i, cv::Mat &_image, int _loop_index, Matrix<double, 8, 1> &_loop_info,
                   vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<float> &_scores,
                   cv::Mat &_local_descriptors, cv::Mat &_global_descriptors, int _height, int _width) {
    time_stamp = _time_stamp;
    index = _index;
    //vio_T_w_i = _vio_T_w_i;
    //vio_R_w_i = _vio_R_w_i;
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
    if (DEBUG_IMAGE)
    {
        image = _image.clone();
        cv::resize(image, thumbnail, cv::Size(80, 60));
    }
    if (_loop_index != -1)
        has_loop = true;
    else
        has_loop = false;
    loop_index = _loop_index;
    loop_info = _loop_info;
    has_fast_point = false;
    sequence = 0;
    keypoints = _keypoints;
    keypoints_norm = _keypoints_norm;
    scores = _scores;
    local_descriptors = _local_descriptors;
    global_descriptors = _global_descriptors;
    height = _height;
    width = _width;
}

void KeyFrame::computeWindow() {
    for(auto & i : point_2d_uv)
    {
        cv::KeyPoint key;
        key.pt = i;
        window_keypoints.push_back(key);
    }
    UltraPoint::Extract(image, point_2d_uv, window_scores, window_local_descriptors);
}

void KeyFrame::computeNew() {
    vector<cv::Point2f> _keypoints;
    std::vector<cv::Point2f> gf_points;

    // extract deep learning keypoints
    SuperPoint::Extract(image, _keypoints, scores, local_descriptors);
    for(auto & i : _keypoints)
    {
        cv::KeyPoint key;
        key.pt = i;
        keypoints.push_back(key);
    }

    // push back the uvs used in vio
    for(auto & i : point_2d_uv)
    {
        cv::KeyPoint key;
        key.pt = i;
        keypoints.push_back(key);
    }
    for(auto & i : window_scores)
    {
        scores.push_back(i);
    }
    cv::hconcat(local_descriptors, window_local_descriptors, local_descriptors);

    for (int i = 0; i < (int)keypoints.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
        keypoints_norm.push_back(tmp_norm);
    }
    NetVLAD::Extract(image, global_descriptors);
}

void KeyFrame::SuperGlueMatcher(vector<cv::Point2f> &matched_2d_old, vector<cv::Point2f> &matched_2d_old_norm, vector<uchar> &status,
                                cv::Mat &local_descriptors_old, vector<float> &scores_old,
                                const vector<cv::KeyPoint> &keypoints_old,
                                const vector<cv::KeyPoint> &keypoints_old_norm,
                                const int height_old, const int width_old) {
    vector<cv::Point2f> _keypoints, _keypoints_old;
    for(auto & i : window_keypoints)
    {
        _keypoints.push_back(i.pt);
    }
    for(auto & i : keypoints_old)
    {
        _keypoints_old.push_back(i.pt);
    }
    vector<int> match_index;
    vector<float> match_score;
    SuperGlue::Match(_keypoints, window_scores, window_local_descriptors, height, width,
                     _keypoints_old, scores_old, local_descriptors_old, height_old, width_old,
                     match_index, match_score
    );
    for (int i : match_index){
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (i >= 0){
            status.push_back(1);
            pt = _keypoints_old[i];
            pt_norm = keypoints_old_norm[i].pt;
        }
        else
            status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }
}

void KeyFrame::FundmantalMatrixRANSAC(const vector<cv::Point2f> &matched_2d_cur_norm,
                                      const vector<cv::Point2f> &matched_2d_old_norm, vector<uchar> &status) {
    int n = (int)matched_2d_cur_norm.size();
    for (int i = 0; i < n; i++)
        status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
    // for (int i = 0; i < matched_3d.size(); i++)
    // 	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
    // printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    int flags = cv::SOLVEPNP_EPNP;
    solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, PNP_INFLATION / max_focallength, 0.99, inliers, flags);

    // if (CV_MAJOR_VERSION < 3)
    //     solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    // else
    // {
    //     if (CV_MINOR_VERSION < 2)
    //         solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
    //     else
    //         solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

    // }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}


bool KeyFrame::findConnection(KeyFrame* old_kf)
{
    TicToc tmp_t;
    //printf("find Connection\n");
    vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
    vector<cv::Point3f> matched_3d;
    vector<double> matched_id;
    vector<uchar> status;

    // re-undistort with the latest intrinsic values
    for (int i = 0; i < (int)point_2d_uv.size(); i++) {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(point_2d_uv[i].x, point_2d_uv[i].y), tmp_p);
        point_2d_norm.push_back(cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z()));
    }
    old_kf->keypoints_norm.clear();
    for (int i = 0; i < (int)old_kf->keypoints.size(); i++) {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(old_kf->keypoints[i].pt.x, old_kf->keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
        old_kf->keypoints_norm.push_back(tmp_norm);
    }

    matched_3d = point_3d;
    matched_2d_cur = point_2d_uv;
    matched_2d_cur_norm = point_2d_norm;
    matched_id = point_id;

    TicToc t_match;
#if 0
    if (DEBUG_IMAGE)
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
    //printf("search by des\n");
    SuperGlueMatcher(matched_2d_old, matched_2d_old_norm, status, old_kf->local_descriptors, old_kf->scores, old_kf->keypoints, old_kf->keypoints_norm, old_kf->height, old_kf->width);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);
    //printf("search by des finish\n");

#if 0
    if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);
	        */

	    }
#endif
    status.clear();
    /*
    FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);
    */
#if 0
    if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
    Eigen::Vector3d PnP_T_old;
    Eigen::Matrix3d PnP_R_old;
    Eigen::Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;
    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
    {
        status.clear();
        PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
        reduceVector(matched_2d_cur, status);
        reduceVector(matched_2d_old, status);
        reduceVector(matched_2d_cur_norm, status);
        reduceVector(matched_2d_old_norm, status);
        reduceVector(matched_3d, status);
        reduceVector(matched_id, status);
#if 1
        if (DEBUG_IMAGE)
        {
            int gap = 10;
	        cv::Mat gap_image(old_kf->image.rows, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f cur_pt = matched_2d_cur[i];
                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
            }
            for(int i = 0; i< (int)matched_2d_old.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (old_kf->image.cols + gap);
                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
            }
            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (old_kf->image.cols + gap);
                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
            cv::Mat notation(50, old_kf->image.cols + gap + old_kf->image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + old_kf->image.cols + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
            cv::vconcat(notation, loop_match_img, loop_match_img);

            /*
            ostringstream path;
            path <<  "/home/tony-ws1/raw_data/loop_image/"
                    << index << "-"
                    << old_kf->index << "-" << "3pnp_match.jpg";
            cv::imwrite( path.str().c_str(), loop_match_img);
            */
            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
            {
                /*
                cv::imshow("loop connection",loop_match_img);
                cv::waitKey(10);
                */
                cv::Mat thumbimage;
                cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
                msg->header.stamp = ros::Time(time_stamp);
                pub_match_img.publish(msg);
            }
        }
#endif
    }

    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
    {
        relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
        relative_q = PnP_R_old.transpose() * origin_vio_R;
        relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
        //printf("PNP relative\n");
        //cout << "pnp relative_t " << relative_t.transpose() << endl;
        //cout << "pnp relative_yaw " << relative_yaw << endl;
        if (abs(relative_yaw) < MAX_THETA_DIFF && relative_t.norm() < MAX_POS_DIFF)
        {

            has_loop = true;
            loop_index = old_kf->index;
            loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
                    relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                    relative_yaw;
            return true;
        }
    }
    //printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
    return false;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = vio_T_w_i;
    R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
    {
        //printf("update loop info\n");
        loop_info = _loop_info;
    }
}
