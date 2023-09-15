/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 */

#pragma once

#include "logging.hpp"
#include "utility.hpp"
#include "dataloader.hpp"

struct PointXYZINS
{
    PCL_ADD_POINT4D;
    PCL_ADD_NORMAL4D;
    union{
        struct 
        {
            float intensity;
            float curvature;
            int segment; // store segmentation result
        };
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZINS,
    (float, intensity, intensity)
    (float, curvature, curvature)
    (int, segment, segment)
)

struct JsonParams
{
    Eigen::MatrixXf intrinsic;
    Eigen::Matrix4f extrinsic, extrinsic_gt;
    std::vector<double> dist;
    std::vector<std::string> img_files, mask_dirs, lidar_files;
    bool is_gt_available = false, is_down_sample = false;
    float search_range_rot, search_range_trans, cluster_tolerance, point_range_top, point_range_bottom, down_sample_voxel;
    int N_FILE, min_plane_point_num, num_thread = 0, search_num;
};

class Calibrator
{
public:
    Calibrator(JsonParams json_params);
    void Calibrate();
    void ProcessPointcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_origin, pcl::PointCloud<PointXYZINS>::Ptr pc);
    double CalScore(Eigen::Matrix4f T);
    void VisualProjection(Eigen::Matrix4f T, std::string save_name, int index);
    void VisualProjectionSegment(Eigen::Matrix4f T, std::string save_name, int index);
    int Segment_pc(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_origin,
                    pcl::PointCloud<pcl::Normal>::Ptr normals,
                    std::vector<pcl::PointIndices>& seg_indices);
    void CalRatio();
    void BruteForceSearch(int rpy_range, float rpy_resolution,int xyz_range, float xyz_resolution);
    void RandomSearch(int search_count, float xyz_range, float rpy_range, int thread_id);
    void RandomSearchThread(int search_count, float xyz_range, float rpy_range);
    bool ProjectOnImage(const Eigen::Vector4f &vec, const Eigen::Matrix4f &T, int &x, int &y, int margin);
    void PrintCurrentError();
    Eigen::Matrix4f GetFinalTransformation();

private:
    JsonParams params_;
    Eigen::Matrix4f extrinsic_ = Eigen::Matrix4f::Identity();
    std::vector<pcl::PointCloud<PointXYZINS>::Ptr> pcs_;
    std::vector<cv::Mat> masks_;
    std::vector<std::vector<int>> mask_point_num_, seg_point_num_;
    std::vector<int> n_mask_, n_seg_;
    // std::vector<std::vector<bool>> mask_valid_;
    std::mutex mtx_;
    Vector<6> best_var_;
    int IMG_H, IMG_W;
    float max_score_ = 2;
    float POINT_PER_PIXEL = 0.05;
    float curvature_max_ = 0;
};