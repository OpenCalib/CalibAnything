#include <chrono> // NOLINT
#include <iostream>
#include <pcl/common/transforms.h>
#include <thread> // NOLINT
#include <time.h>
#include <jsoncpp/json/json.h>

#include "calibration.hpp"

void readConfig(std::string json_file, JsonParams &json_params)
{
    auto n = json_file.find_last_of('/');
    std::string data_root;
    if (n != std::string::npos)
    {
        data_root = json_file.substr(0, n + 1);
    }
    else{
        data_root = "";
    }

    Json::Reader reader;
	Json::Value root;
 
	std::ifstream ifs(json_file, std::ios::binary); 
	if (!ifs.is_open())
	{
		std::cout << "Can't open file" << json_file << std::endl;
		exit(1);
	}
 
	if (reader.parse(ifs, root))
	{
        // read K
        json_params.intrinsic.resize(root["cam_K"]["rows"].asInt(), root["cam_K"]["cols"].asInt());
        for (int i = 0; i < root["cam_K"]["rows"].asInt(); i++)
        {
            for (int j = 0; j < root["cam_K"]["cols"].asInt(); j++)
            {
                json_params.intrinsic(i, j) = root["cam_K"]["data"][i][j].asFloat();
            }
        }

        // read dist
        for (int i = 0; i < root["cam_dist"]["cols"].asInt(); i++)
        {
            json_params.dist.push_back(root["cam_dist"]["data"][i].asDouble());
        }

        // read T
        for (int i = 0; i < root["T_lidar_to_cam"]["rows"].asInt(); i++)
        {
            for (int j = 0; j < root["T_lidar_to_cam"]["cols"].asInt(); j++)
            {
                json_params.extrinsic(i, j) = root["T_lidar_to_cam"]["data"][i][j].asFloat();
            }
        }

        // read T_gt
        if (root["T_lidar_to_cam_gt"]["available"].asBool()){
            json_params.is_gt_available = true;
            for (int i = 0; i < root["T_lidar_to_cam_gt"]["rows"].asInt(); i++)
            {
                for (int j = 0; j < root["T_lidar_to_cam_gt"]["cols"].asInt(); j++)
                {
                    json_params.extrinsic_gt(i, j) = root["T_lidar_to_cam_gt"]["data"][i][j].asFloat();
                }
            }
        }

        // read data dirs
        std::string img_folder = data_root + root["img_folder"].asString();
        std::string mask_folder = data_root + root["mask_folder"].asString();
        std::string pc_folder = data_root + root["pc_folder"].asString();
        for (int i = 0; i < root["file_name"].size(); i++)
        {
			std::string item_name = root["file_name"][i].asString();
            json_params.img_files.push_back(img_folder + "/" + item_name + root["img_format"].asString());
            json_params.mask_dirs.push_back(mask_folder + "/" + item_name);
            json_params.lidar_files.push_back(pc_folder + "/" + item_name + root["pc_format"].asString());
        }
        json_params.N_FILE = root["file_name"].size();

        // read params
        json_params.search_range_rot = root["params"]["search_range"]["rot_deg"].asFloat();
        json_params.search_range_trans = root["params"]["search_range"]["trans_m"].asFloat();
        json_params.min_plane_point_num = root["params"]["min_plane_point_num"].asInt();
        json_params.cluster_tolerance = root["params"]["cluster_tolerance"].asFloat();
        json_params.point_range_bottom = root["params"]["point_range"]["bottom"].asFloat();
        json_params.point_range_top = root["params"]["point_range"]["top"].asFloat();
        json_params.search_num = root["params"]["search_num"].asInt();
        if(root["params"]["thread"]["is_multi_thread"].asBool()){
            json_params.num_thread = root["params"]["thread"]["num_thread"].asInt();
        }
        if (root["params"]["down_sample"]["is_valid"].asBool())
        {
            json_params.is_down_sample = true;
            json_params.down_sample_voxel = root["params"]["down_sample"]["voxel_m"].asFloat();
        }

        std::cout << "Reading json file complete!" << std::endl;
    }

    ifs.close();
}

void SaveExtrinsic(Eigen::Matrix4f T)
{
    std::string file_name = "extrinsic.txt";
    
    std::ofstream ofs(file_name);
    if (!ofs.is_open())
    {
        std::cerr << "open file " << file_name << " failed. Cannot write calib result." << std::endl;
        exit(1);
    }
    ofs << "Estimate extrinsic Matrix: " << std::endl;
    ofs << "[[" << T(0, 0) << "," << T(0, 1) << "," << T(0, 2) << "," << T(0, 3) << "]," << std::endl
        << "[" << T(1, 0) << "," << T(1, 1) << "," << T(1, 2) << "," << T(1, 3) << "]," << std::endl
        << "[" << T(2, 0) << "," << T(2, 1) << "," << T(2, 2) << "," << T(2, 3) << "]," << std::endl
        << "[" << T(3, 0) << "," << T(3, 1) << "," << T(3, 2) << "," << T(3, 3) << "]]" << std::endl;

    ofs.close();

    std::cout << "Calibration result was saved to file calib_result.txt" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./bin/run_lidar2camera config_file\n"
                     "example:\n\t"
                     "./bin/run_lidar2camera data/kitti.json\n" << std::endl;
        return 0;
    }

    std::string json_file = argv[1];
    JsonParams json_params;
    readConfig(json_file, json_params);

    auto time_begin = std::chrono::steady_clock::now();
    Calibrator calibrator(json_params);
    calibrator.Calibrate();
    Eigen::Matrix4f refined_extrinsic = calibrator.GetFinalTransformation();
    SaveExtrinsic(refined_extrinsic);
    auto time_end = std::chrono::steady_clock::now();
    std::cout << "Total calib time: "
                << std::chrono::duration<double>(time_end - time_begin).count()
                << "s" << std::endl;
                
    return 0;
}
