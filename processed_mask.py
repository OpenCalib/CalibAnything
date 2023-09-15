'''
Usage:
    python processed_mask.py mask_dir processed_mask_dir
Example:
    python processed_mask.py data/nuscenes/masks data/nuscenes/processed_masks
'''

import numpy as np
import cv2
import os
import time
from shutil import copyfile
import argparse

def main(input_dir:str, output_dir:str):

    masks_list = os.listdir(input_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        print("Error: output_dir already exist!")
        return

    for mask_name in masks_list:

        input_dir_sub = input_dir + '/' + mask_name
        output_dir_sub = output_dir + '/' + mask_name + '/'       
        os.makedirs(output_dir_sub)

        mask_files = os.listdir(input_dir_sub)
        n = 0
        for file in mask_files:
            if file[-4:] != '.png': continue
            mask_ori = cv2.imread(input_dir_sub + '/' + file)
            H, W, _ = mask_ori.shape
            mask_ori = mask_ori > 128


            mask_ori = np.asarray(mask_ori[:, :, 0], dtype=np.double)
            n_white = np.sum(mask_ori)
            gx, gy = np.gradient(mask_ori)
            temp_edge = gy * gy + gx * gx


            temp_edge[temp_edge != 0.0] = 1
            
            if n_white < 0.02 * H * W:
                copyfile(input_dir_sub + '/' + file, output_dir_sub + '/' + file)            
            else:
                mask_new1 = np.zeros(mask_ori.shape, dtype=bool)
                margin_inside = int(30 + H * W / n_white)
                for i in range(H):
                    for j in range(W):
                        if temp_edge[i][j] != 0:
                            left = max(j - margin_inside, 0)
                            right = min(j + margin_inside, W - 1)
                            top = max(i - margin_inside,0)
                            bottom = min(i + margin_inside, H - 1)
                            mask_new1[top:bottom, left:right] = 1

                mask_out = np.zeros(mask_ori.shape)
                mask_out[np.logical_and(mask_ori, mask_new1)] = 255
                np.asarray(mask_out, dtype=np.uint8)
                cv2.imwrite(output_dir_sub + file, mask_out)
                n += 1
        
        print("processed {} images".format(n))      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='processed_mask',
                    description='cut_large_mask')
    parser.add_argument('-i', '--input', type=str, help='origin masks folder path')
    parser.add_argument('-o', '--output', type=str, help='processed masks folder path')
    args = parser.parse_args()
    main(args.input, args.output)