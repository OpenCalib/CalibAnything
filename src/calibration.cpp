#include "calibration.hpp"

cv::Mat color_bar = cv::Mat(1, 13 * 3 * 3, CV_8UC3);
unsigned char *pBar = color_bar.data;
void Create_ColorBar()
{
    int H[13] = {180, 120, 60, 160, 100, 40, 150, 90, 30, 140, 80, 20, 10};
    int S[3] = {255, 100, 30};
    int V[3] = {255, 180, 90};

    cv::Mat color = cv::Mat(1, 13 * 3 * 3, CV_8UC3);
    unsigned char *pColor = color.data;

    int h = 0, s = 0, v = 0;
    for (int ba = 0; ba < 13 * 3 * 3; v++, s++, h++, ba++)
    {
        if (h == 13)
            h = 0;
        if (s == 3 * 13)
            s = 0;
        if (v == 3 * 13 * 3)
            v = 0;
        pColor[ba * 3 + 0] = H[h];
        pColor[ba * 3 + 1] = S[s / 13];
        pColor[ba * 3 + 2] = V[v / 13 / 3];
    }
    cv::cvtColor(color, color_bar, CV_HSV2BGR);
}


Calibrator::Calibrator(JsonParams json_params)
{
    params_ = json_params;
    extrinsic_ = json_params.extrinsic;
    std::cout << "----------Start processing data----------" << std::endl;

    // load image
    cv::Mat img = cv::imread(params_.img_files[0]);
    if (!img.data)
    {  
        std::cout << "Can not read " << params_.img_files[0] << std::endl;  
        exit(1);  
    }
    IMG_H = img.rows;
    IMG_W = img.cols;

    for (int i = 0; i < params_.N_FILE; i++)
    {
        std::cout << "Processing data " << i + 1 << ":" << std::endl;
        std::vector<int> mask_point_num;
        cv::Mat masks = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC4);
        DataLoader::LoadMaskFile(params_.mask_dirs[i], params_.intrinsic, params_.dist, masks, mask_point_num);
        masks_.push_back(masks);
        mask_point_num_.push_back(mask_point_num);
        n_mask_.push_back(mask_point_num.size());

        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_origin(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<PointXYZINS>::Ptr pc(new pcl::PointCloud<PointXYZINS>);
        DataLoader::LoadLidarFile(params_.lidar_files[i], pc_origin);
        ProcessPointcloud(pc_origin, pc);
        pcs_.push_back(pc);
    }
    Create_ColorBar();
}

void Calibrator::ProcessPointcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc_origin, pcl::PointCloud<PointXYZINS>::Ptr pc)
{
    // pre-filter points by initial extrinsic
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    int margin = 300;
    float intensity_max = 1;
    std::vector<Vector<6>> vars;
    if(params_.search_range_rot > 10 || params_.search_range_trans > 1){
        std::cout << "Search range too large. Don't support!" << std::endl;
    }
    Util::GenVars(1, DEG2RAD(params_.search_range_rot), 1, params_.search_range_trans, vars);
    for (const auto src_pt : pc_origin->points)
    {
        if (!std::isfinite(src_pt.x) || !std::isfinite(src_pt.y) ||
            !std::isfinite(src_pt.z))
            continue;
        Eigen::Vector4f vec;
        vec << src_pt.x, src_pt.y, src_pt.z, 1;
        int x, y;
        for(Vector<6> var : vars){
            Eigen::Matrix4f extrinsic = extrinsic_ * Util::GetDeltaT(var);
            if (ProjectOnImage(vec, extrinsic, x, y, margin))
            {
                intensity_max = MAX(intensity_max, src_pt.intensity);
                pc_filtered->points.push_back(src_pt);
                break;
            }
        }
    }
    std::cout << "Point cloud num: " << pc_filtered->points.size() << std::endl;
    // pc_filtered->height = 1;
    // pc_filtered->width = pc_filtered->points.size();
    // pcl::io::savePCDFileASCII("cloud_filtered.pcd", *pc_filtered);

    // downsample
    if(params_.is_down_sample){
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tmp_ds(new pcl::PointCloud<pcl::PointXYZI>);
        const double leafSize = params_.down_sample_voxel;
        pcl::VoxelGrid<pcl::PointXYZI> filter_map;
        filter_map.setLeafSize(leafSize, leafSize, leafSize);
        cloud_downsampled->clear();

        pcl::octree::OctreePointCloud<pcl::PointXYZI> octree{1250.0 * leafSize};
        octree.setInputCloud(pc_filtered);
        octree.addPointsFromInputCloud();
        for (auto it = octree.leaf_depth_begin(); it != octree.leaf_depth_end(); ++it) {
            auto ids = it.getLeafContainer().getPointIndicesVector();
            cloud_tmp->clear();
            for (auto id : ids) {
                cloud_tmp->push_back(octree.getInputCloud()->points[id]);
            }
            filter_map.setInputCloud(cloud_tmp);
            filter_map.filter(*cloud_tmp_ds);
            *cloud_downsampled += *cloud_tmp_ds;
        }
        pc_filtered = cloud_downsampled;
        std::cout << "Points num after downsample: " << pc_filtered->points.size() << std::endl;
    }

    // segment pc
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    std::vector<pcl::PointIndices> seg_indices;
    int n_seg = Segment_pc(pc_filtered, normals, seg_indices);
    n_seg_.push_back(n_seg);
    std::cout << "Extract " << n_seg << " segments from point cloud." << std::endl;

    // constuct new type pc
    for (unsigned i = 0; i < pc_filtered->size(); i++)
    {
        PointXYZINS pt;
        pt.x = (*pc_filtered)[i].x;
        pt.y = (*pc_filtered)[i].y;
        pt.z = (*pc_filtered)[i].z;
        pt.intensity = (*pc_filtered)[i].intensity / intensity_max;
        pt.normal_x = (*normals)[i].normal_x;
        pt.normal_y = (*normals)[i].normal_y;
        pt.normal_z = (*normals)[i].normal_z;
        pt.curvature = (*normals)[i].curvature;
        curvature_max_ = MAX(curvature_max_, pt.curvature);
        // std::cout << pt.curvature << std::endl;
        pt.segment = -1;
        pc->points.push_back(pt);
    }

    // curvature_max_ /= 2;

    for (int i = 0; i < n_seg; i++)
    {
        for (int index : seg_indices[i].indices)
        {
            (*pc)[index].segment = i;
        }
    }
}

int Calibrator::Segment_pc(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                            pcl::PointCloud<pcl::Normal>::Ptr normals,
                            std::vector<pcl::PointIndices> &seg_indices)
{   
    // compute_normals
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> norm_est;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZI>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(40);
    // norm_est.setRadiusSearch(5);
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);

    // plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr indices_plane(new pcl::PointIndices);
    pcl::SACSegmentationFromNormals<pcl::PointXYZI, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.2);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(3000);
    seg.setDistanceThreshold(0.2);
    seg.setInputCloud(cloud);
    seg.setInputNormals(normals);
    seg.segment(*indices_plane, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZI> extract(true);
    extract.setInputCloud(cloud);
    pcl::PointIndices::Ptr indices_notplane(new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>);
    int plane_size = indices_plane->indices.size();
    pcl::PointIndices::Ptr indices_plane_all(new pcl::PointIndices);
    std::vector<int> seg_point_num;
    while (plane_size > params_.min_plane_point_num)
    {
        std::cout << "Plane points: " << plane_size << std::endl;
        seg_indices.push_back(*indices_plane);
        seg_point_num.push_back(plane_size);
        indices_plane_all->indices.insert(indices_plane_all->indices.end(), indices_plane->indices.begin(), indices_plane->indices.end());
        extract.setIndices(indices_plane_all);
        extract.filter(*cloud_out);
        extract.getRemovedIndices(*indices_notplane);
        seg.setIndices(indices_notplane);
        seg.segment(*indices_plane, *coefficients);
        plane_size = indices_plane->indices.size();
    }
    std::cout << "Plane points < " << params_.min_plane_point_num << ", stop extracting plane." << std::endl;

    // euclidean cluster extraction
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    std::vector<pcl::PointIndices> eu_cluster_indices;
    ec.setClusterTolerance(params_.cluster_tolerance);
    ec.setMaxClusterSize(10000);
    ec.setMinClusterSize(50);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.setIndices(indices_notplane);
    ec.extract(eu_cluster_indices);
    std::cout << "Euclidean cluster number: " << eu_cluster_indices.size() << std::endl;


    seg_indices.insert(seg_indices.end(), eu_cluster_indices.begin(), eu_cluster_indices.end());
    for (auto it = eu_cluster_indices.begin(); it != eu_cluster_indices.end(); it++)
    {
        seg_point_num.push_back((*it).indices.size());
    }

    // visualize clustering result
    // int j = 0;
    // pcl::PCDWriter writer;
    // for (const auto& cluster : seg_indices)
    // {
    //     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
    //     for (const auto& idx : cluster.indices) {
    //         cloud_cluster->push_back((*cloud)[idx]);
    //     } //*
    //     cloud_cluster->width = cloud_cluster->size ();
    //     cloud_cluster->height = 1;
    //     cloud_cluster->is_dense = true;

    //     std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
    //     std::stringstream ss;
    //     ss << std::setw(4) << std::setfill('0') << j;
    //     writer.write<pcl::PointXYZI> ("cloud_cluster_" + ss.str () + ".pcd", *cloud_cluster, false); //*
    //     j++;
    // }

    seg_point_num_.push_back(seg_point_num);
    return seg_indices.size();
}

void Calibrator::Calibrate()
{
    CalRatio();
    std::cout << "----------Start calibration----------" << std::endl;
    VisualProjectionSegment(extrinsic_, "init_proj_seg.png", 0);
    VisualProjection(extrinsic_, "init_proj.png", 0);
    if (params_.is_gt_available){
        VisualProjectionSegment(params_.extrinsic_gt, "gt_proj_seg.png", 0);
        VisualProjection(params_.extrinsic_gt, "gt_proj.png", 0);
    }
    max_score_ = CalScore(extrinsic_);
    float rot_init = params_.search_range_rot + 0.5;
    float trans_init = params_.search_range_trans + 0.05;
    do{
        best_var_ << 0, 0, 0, 0, 0, 0;
        if(params_.num_thread != 0)
            RandomSearchThread(params_.search_num, trans_init, DEG2RAD(rot_init));
        else 
            RandomSearch(params_.search_num, trans_init, DEG2RAD(rot_init), 0);
        extrinsic_ *= Util::GetDeltaT(best_var_);
        rot_init /= 2;
        trans_init /= 1.5;
    } while (rot_init > 0.3);
    std::cout << "----------Calibration complete----------" << std::endl;
    if(params_.is_gt_available)
    {
        PrintCurrentError();
    }
    VisualProjectionSegment(extrinsic_, "refined_proj_seg.png", 0);
    VisualProjection(extrinsic_, "refined_proj.png", 0);
}

double Calibrator::CalScore(Eigen::Matrix4f T)
{
    double score = 0;
    float normal_sum = 0, intensity_sum = 0, segment_sum = 0;
    for (int f = 0; f < params_.N_FILE; f++)
    {
        int n_mask = n_mask_[f];
        pcl::PointCloud<PointXYZINS>::Ptr pc(pcs_[f]);
        cv::Mat masks = masks_[f];
        std::vector<std::vector<float>> mask_normal(n_mask);
        std::vector<std::vector<float>> mask_intensity(n_mask);
        std::vector<std::unordered_map<int, int>> mask_segment(n_mask);
        int n_bottom = 0, n_top = 0;
        for (const auto &src_pt : pc->points)
        {
            Eigen::Vector4f vec;
            vec << src_pt.x, src_pt.y, src_pt.z, 1;
            int x, y;
            if (ProjectOnImage(vec, T, x, y, 0))
            {
                if(y > (params_.point_range_bottom - 0.1) * IMG_H && y < params_.point_range_bottom * IMG_H){n_bottom++;}
                // else if(y < (params_.point_range_top - 0.2) * IMG_H){n_top++;}
                cv::Vec4b mask_id = masks.at<cv::Vec4b>(y, x);
                for (int c = 0; c < 4; c++)
                {
                    if (mask_id[c] != 0)
                    {
                        mask_normal[mask_id[c] - 1].push_back(src_pt.normal_x);
                        mask_normal[mask_id[c] - 1].push_back(src_pt.normal_y);
                        mask_normal[mask_id[c] - 1].push_back(src_pt.normal_z);
                    
                        mask_intensity[mask_id[c] - 1].push_back(src_pt.intensity);
                        if (src_pt.segment != -1)
                        {
                            mask_segment[mask_id[c] - 1][src_pt.segment]++;
                        }
                    }
                    else
                        break;
                }
            }
        }
        // std::cout << n_bottom << std::endl;

        if (n_bottom < 0.1 * POINT_PER_PIXEL * 0.1 * IMG_H * IMG_W ) // || n_top > 0.05 * POINT_PER_PIXEL * (params_.point_range_top - 0.2) * IMG_H * IMG_W)
        {
            // std::cout << "Not enough points on the bottom of the image." << std::endl;
            score += 2;
            continue;
        }
        // calculate consistency

        std::vector<float> normal_scores, intensity_scores, segment_scores;
        std::vector<float> weight_normal, weight_intensity, weight_seg;
        int min_pixel = IMG_H * IMG_W / 1200;
        min_pixel = MIN(min_pixel, 2000);
        for (int i = 0; i < n_mask_[f]; i++)
        {
            // filter masks with too few points
            if (mask_point_num_[f][i] < min_pixel)
                continue;
            int num_base = mask_point_num_[f][i] * POINT_PER_PIXEL;
            int num_inside = mask_intensity[i].size();
            // std::cout << num_base << std::endl;
            // std::cout << num_inside << std::endl;

            float weight = num_inside;
            if (num_inside < 0.1 * num_base || num_inside < 10)
                continue;     

            
            // float adjust = 1 - 0.5 * pow(num_inside, -0.5);
            float adjust = 1;

            // normal consistency
            Eigen::MatrixXf normals;
            normals = Eigen::Map<Eigen::MatrixXf, Eigen::Unaligned>(mask_normal[i].data(), 3, num_inside);
            float normal_sim = (normals.transpose() * normals).array().abs().mean(); 
            normal_scores.push_back(normal_sim * adjust);
            weight_normal.push_back(weight);

            // intensity consistency
            float intensity_sim = (1 - Util::Std(mask_intensity[i]));
            intensity_scores.push_back(intensity_sim * adjust);
            weight_intensity.push_back(weight);

            // segment consistency
            if (mask_segment[i].size() > 0)
            {
                std::vector<int> seg_ratio;
                for (auto it = mask_segment[i].begin(); it != mask_segment[i].end(); it++)
                {
                    seg_ratio.push_back(it->second);
                }
                sort(seg_ratio.begin(), seg_ratio.end(),std::greater<int>());
                float k = 1;
                int sum = 0;
                float segment_sim = 0;
                for (auto it = seg_ratio.begin(); it != seg_ratio.end(); it++)
                {
                    segment_sim += *it * k;
                    k *= 0.5;
                    sum += *it;
                }
                // if(seg_ratio[0] * 1.0 / sum < 0.2)
                //     continue;
                // segment_sim = segment_sim / sum * (1 + 0.05 * log10(sum));
                segment_sim = segment_sim / sum;
                segment_scores.push_back(segment_sim);
                weight_seg.push_back(sum);
            }
        }
        if (normal_scores.size() == 0 || intensity_scores.size() == 0 || segment_scores.size() == 0)
        {
            //  std::cout << "Not enough points on masks." << std::endl;
            score += 2;
            continue;
        }
        // std::cout << normal_scores.size() << " " << intensity_scores.size() << segment_scores.size() << std::endl;

        float normal_score, intensity_score, segment_score;
        normal_score = Util::WeightMean(normal_scores, weight_normal);
        intensity_score = Util::WeightMean(intensity_scores, weight_intensity);
        segment_score = Util::WeightMean(segment_scores, weight_seg);
        normal_sum += normal_score;
        intensity_sum += intensity_score;
        segment_sum += segment_score;

        score += 2 - 0.3 * normal_score - 0.2 * intensity_score - 0.5 * segment_score 
            - 0.0001 * normal_scores.size();

        // std::cout << "score: " << normal_score << " " << intensity_score << " " << segment_score <<
        //     " " << score << std::endl;
    }
    score = score / params_.N_FILE;
    // std::cout << "score: " << normal_sum<< " " << intensity_sum << " " << segment_sum <<
    // " " << score << std::endl;
    return score;
}

void Calibrator::BruteForceSearch(int rpy_range, float rpy_resolution, int xyz_range, float xyz_resolution)
{
    Vector<6> best_var;
    float score;
    std::vector<Vector<6>> vars;
    Util::GenVars(rpy_range, rpy_resolution, xyz_range, xyz_resolution, vars);
    for (auto var : vars)
    {
        Eigen::Matrix4f T = extrinsic_ * Util::GetDeltaT(var);
        float score = CalScore(T);
        if (score < max_score_)
        {
            max_score_ = score;
            for (size_t k = 0; k < 6; k++)
            {
                best_var[k] = var(k);
            }

            std::cout << "Loss decreases to: " << max_score_ << ", "
                    << "var:" << best_var[0] << " " << best_var[1] << " " << best_var[2]
                    << " " << best_var[3] << " " << best_var[4] << " " << best_var[5]
                    << std::endl;
        }
    }    
    std::cout << "best var:" << best_var[0] << " " << best_var[1] << " " << best_var[2] << " " << best_var[3] << " "
              << best_var[4] << " " << best_var[5] << std::endl;
    Eigen::Matrix4f deltaT = Util::GetDeltaT(best_var);
    extrinsic_ *= deltaT;
}

void Calibrator::RandomSearchThread(int search_count, float xyz_range, float rpy_range)
{
    std::vector<std::thread> threads;
    for (int i = 0; i < params_.num_thread; i++)
    {
        threads.push_back(std::thread(&Calibrator::RandomSearch, this, int(search_count / params_.num_thread), xyz_range, rpy_range, i));
    }
    for (int i = 0; i < params_.num_thread; i++)
    {
        if(threads[i].joinable()){
            threads[i].join();
        }
    }
}

void Calibrator::RandomSearch(int search_count, float xyz_range, float rpy_range, int thread_id)
{
    if(thread_id == 0){
        std::cout << "Start random search around [-" << RAD2DEG(rpy_range) << "," << RAD2DEG(rpy_range) << "] degree and [-"
              << xyz_range << "," << xyz_range << "] m" << std::endl;
    }
    Vector<6> var, best_var;
    double max_score = max_score_;
    std::default_random_engine generator((clock() - time(0)) /
                                         (double)CLOCKS_PER_SEC + thread_id);
    std::uniform_real_distribution<double> distribution_xyz(-xyz_range, xyz_range);
    std::uniform_real_distribution<double> distribution_rpy(-rpy_range, rpy_range);
    for (int i = 0; i < search_count; i++)
    {
        var[0] = distribution_rpy(generator);
        var[1] = distribution_rpy(generator);
        var[2] = distribution_rpy(generator);
        var[3] = distribution_xyz(generator);
        var[4] = distribution_xyz(generator);
        var[5] = distribution_xyz(generator);
        Eigen::Matrix4f T = extrinsic_ * Util::GetDeltaT(var);
        float score = CalScore(T);
        if (score < max_score)
        {
            max_score = score;
            best_var = var;
        }
    }
    std::unique_lock<std::mutex> lg(mtx_);
    if(max_score < max_score_)
    {
        std::cout << "Thread " << thread_id << " Loss decreases to: " << max_score << ", "
                      << "var:" << best_var[0] << " " << best_var[1] << " " << best_var[2]
                      << " " << best_var[3] << " " << best_var[4] << " " << best_var[5]
                      << std::endl;
        max_score_ = max_score;
        best_var_ = best_var;
    }
}

void Calibrator::PrintCurrentError()
{
    Eigen::Matrix4f error_T = params_.extrinsic_gt * extrinsic_.inverse();
    std::cout << "Error(roll pitch yaw x y z): " << RAD2DEG(Util::GetRoll(error_T)) << " " << RAD2DEG(Util::GetPitch(error_T)) << " " << RAD2DEG(Util::GetYaw(error_T))
              << " " << Util::GetX(error_T) << " " << Util::GetY(error_T) << " " << Util::GetZ(error_T) << std::endl;
}

Eigen::Matrix4f Calibrator::GetFinalTransformation()
{
    return extrinsic_;
}

void Calibrator::VisualProjection(Eigen::Matrix4f T, std::string save_name, int index)
{
    cv::Mat img_color = cv::imread(params_.img_files[index]);
    if(!img_color.data)
    {  
        std::cout << "Can not read " << params_.img_files[index] << std::endl;  
        exit(1);  
    }
    if(params_.intrinsic.cols() == 3)
    {
        Util::UndistImg(img_color, params_.intrinsic, params_.dist);
    }
    std::vector<cv::Point2f> lidar_points;
    for (const auto &src_pt : pcs_[index]->points)
    {
        Eigen::Vector4f vec;
        vec << src_pt.x, src_pt.y, src_pt.z, 1;
        int x, y;
        if (ProjectOnImage(vec, T, x, y, 0))
        {
            // if (src_pt.intensity > 0.4) {
                cv::Point2f lidar_point(x, y);
                lidar_points.push_back(lidar_point);
            // }
        }
    }
    // std::cout << lidar_points.size() << std::endl;
    for (cv::Point point : lidar_points)
    {
        cv::circle(img_color, point, 3, cv::Scalar(0, 255, 0), -1, 0);
    }
    cv::imwrite(save_name, img_color);
    std::cout << "Image saved: " << save_name << std::endl;
}

void Calibrator::CalRatio()
{
    std::vector<cv::Point2f> lidar_points;
    int point_num = 0;
    assert(params_.point_range_top < params_.point_range_bottom);
    int top = params_.point_range_top * IMG_H;
    int bottom = params_.point_range_bottom * IMG_H;
    for (const auto &src_pt : pcs_[0]->points)
    {
        Eigen::Vector4f vec;
        vec << src_pt.x, src_pt.y, src_pt.z, 1;
        int x, y;
        if (ProjectOnImage(vec, extrinsic_, x, y, 0))
        {
            if(y > top && y < bottom){
                point_num++;
            }
        }
    }
    int area = (params_.point_range_bottom - params_.point_range_top) * IMG_H * IMG_W;
    POINT_PER_PIXEL = point_num * 1.0 / area;

    std::cout << "Estimated point number per pixel: " << POINT_PER_PIXEL << std::endl;
}

void Calibrator::VisualProjectionSegment(Eigen::Matrix4f T, std::string save_name, int index)
{
    cv::Mat img_color2 = cv::imread(params_.img_files[index]);
    if (!img_color2.data)
    {  
        std::cout << "Can not read " << params_.img_files[index] << std::endl;  
        exit(1);  
    }
    cv::Mat img_color;
    cv::resize(img_color2, img_color, cv::Size(IMG_W, IMG_H));  
    if(params_.intrinsic.cols() == 3)
    {
        Util::UndistImg(img_color, params_.intrinsic, params_.dist);
    }
    std::vector<std::vector<cv::Point2f>> lidar_points(n_seg_[index]);
    for (const auto &src_pt : pcs_[index]->points)
    {
        Eigen::Vector4f vec;
        vec << src_pt.x, src_pt.y, src_pt.z, 1;
        int x, y;
        if (ProjectOnImage(vec, T, x, y, 0))
        {
            int seg = src_pt.segment;
            if (seg != -1 && seg != 0)
            {
                cv::Point2f lidar_point(x, y);
                lidar_points[seg].push_back(lidar_point);
            }
        }
    }

    for (int i = 0; i < n_seg_[index]; i++)
    {
        cv::Vec3b color = color_bar.at<cv::Vec3b>(0, i);
        for (cv::Point point : lidar_points[i])
        {
            cv::circle(img_color, point, 3, cv::Scalar(color[0], color[1], color[2]), -1, 0);
        }
    }
    cv::imwrite(save_name, img_color);
    std::cout << "Image saved: " << save_name << std::endl;
}

bool Calibrator::ProjectOnImage(const Eigen::Vector4f &vec, const Eigen::Matrix4f &T, int &x, int &y, int margin)
{
    Eigen::Vector3f vec2;
    if (params_.intrinsic.cols() == 4)
    {
        vec2 = params_.intrinsic * T * vec;
    }
    else
    {
        Eigen::Vector4f cam_point = T * vec;
        Eigen::Vector3f cam_vec;
        cam_vec << cam_point(0), cam_point(1), cam_point(2);
        vec2 = params_.intrinsic * cam_vec;
    }
    if (vec2(2) <= 0)
        return false;
    x = (int)cvRound(vec2(0) / vec2(2));
    y = (int)cvRound(vec2(1) / vec2(2));
    if (x >= -margin && x < IMG_W + margin && y >= -margin && y < IMG_H + margin)
        return true;
    return false;
}