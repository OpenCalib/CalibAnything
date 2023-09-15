## CalibAnything

This package provides an automatic and target-less LiDAR-camera extrinsic calibration method using Segment Anything Model. The related paper is [Calib-Anything: Zero-training LiDAR-Camera Extrinsic Calibration Method Using Segment Anything](https://arxiv.org/abs/2306.02656). For more calibration codes, please refer to the link <a href="https://github.com/PJLab-ADG/SensorsCalibration" title="SensorsCalibration">SensorsCalibration</a>.

## Prerequisites
- pcl 1.10
- opencv
- eigen 3

## Compile
```shell
git clone https://github.com/OpenCalib/CalibAnything.git
cd CalibAnything
# mkdir build
mkdir -p build && cd build
# build
cmake .. && make
```

## Run Example
We provide examples of two dataset. You can download the processed data at [Google Drive](https://drive.google.com/file/d/1OCtbIGilLOBnHzY5VNHqRZzbxXj3xiXc/view?usp=drive_link) or [BaiduNetDisk](https://pan.baidu.com/s/1qAt7nYw5hYoJ1qrH0JosaQ?pwd=417d):
```
# baidunetdisk
Link: https://pan.baidu.com/s/1qAt7nYw5hYoJ1qrH0JosaQ?pwd=417d 
Code: 417d
```

Run the command:
```shell
cd CalibAnything
./bin/run_lidar2camera ./data/kitti/calib.json # kitti dataset
./bin/run_lidar2camera ./data/nuscenes/calib.json # nuscenes dataset
```

## Test your own data

### Data collection

- Several pairs of time synchronized RGB images and LiDAR point cloud (intensity is needed). One pair of data can also be used to calibrate, but the results may be ubstable.
- The intrinsic of the camera and the initial guess of the extrinsic.

### Preprocessing

#### Generate masks

Follow the instructions in [Segment Anything](https://github.com/facebookresearch/segment-anything) and generate masks of your image.

1. First download a model checkpoint. You can choose [vit-l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth).

2. Install SAM
```shell
# environment: python>=3.8, pytorch>=1.7, torchvision>=0.8

git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

3. Run
```shell
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>

# example(recommend parameter)
python scripts/amg.py --checkpoint sam_vit_l_0b3195.pth --model-type vit_l --input ./data/kitti/000000/images/  --output ./data/kitti/000000/masks/ --stability-score-thresh 0.9 --box-nms-thresh 0.5 --stability-score-offset 0.9
```

#### Data folder
The hierarchy of your folders should be formed as:
```
YOUR_DATA_FOLDER
├─calib.json
├─pc
|   ├─000000.pcd
|   ├─000001.pcd
|   ├─...
├─images
|   ├─000000.png
|   ├─000001.png
|   ├─...
├─masks
|   ├─000000
|   |   ├─000.png
|   |   ├─001.png
|   |   ├─...
|   ├─000001
|   ├─...

```

#### Processed masks

For large masks, we only use part of it near the edge.
```shell
python processed_mask.py -i <YOUR_DATA_FOLDER>/masks/ -o <YOUR_DATA_FOLDER>/processed_masks/
```

#### Edit the json file

<details><summary>Content description</summary>

- `cam_K`: camera intrinsic matrix
- `cam_dist`: camera distortion coefficient. `[k1, k2, p1, p2, p3, ...]`, use the same order as [opencv](https://amroamroamro.github.io/mexopencv/matlab/cv.initUndistortRectifyMap.html)
- `T_lidar_to_cam`: initial guess of the extrinsic
- `T_lidar_to_cam_gt`: ground-truth of the extrinsic (Used to calculate error. If not provided, set "available" to false)
- `img_folder`: the path to images
- `mask_folder`: the path to masks
- `pc_folder`: the path to point cloud
- `img_format`: the suffix of the image
- `pc_format`: the suffix of the point cloud (support pcd or kitti bin)
- `file_name`: the name of the input images and point cloud
- `min_plane_point_num`: the minimum number of point in plane extraction
- `cluster_tolerance`: the spatial cluster tolerance in euclidean cluster (set larger if the point cloud is sparse, such as the 32-beam LiDAR)
- `search_num`: the number of search times
- `search_range`: the search range for rotation and translation
- `point_range`: the approximate height range of the point cloud projected onto the image (the top of the image is 0.0 and the bottom of the image is 1.0)
- `down_sample`: the point cloud downsample voxel size (if don't need downsample, set the "is_valid" to false)
- `thread`: the number of thread to reduce calibration time
</details>

### Calibration
```shell
./bin/run_lidar2camera <path-to-json-file>
```

## Output
- initial projection: `init_proj.png`, `init_proj_seg.png`
- gt projection: `gt_proj.png`, `gt_proj_seg.png`
- refined projection: `refined_proj.png`, `refined_proj_seg.png`
- refined extrinsic: `extrinsic.txt`

## Citation
If you find this project useful in your research, please consider cite:
```
@misc{luo2023calibanything,
      title={Calib-Anything: Zero-training LiDAR-Camera Extrinsic Calibration Method Using Segment Anything}, 
      author={Zhaotong Luo and Guohang Yan and Yikang Li},
      year={2023},
      eprint={2306.02656},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
