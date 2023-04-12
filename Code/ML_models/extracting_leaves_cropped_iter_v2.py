# Need to make a /resize folder in output dir

# python /data/botml/leaf_dimension_classifier/code/extracting_leaves_cropped_iter_v2.py "/data/botml/leaf_dimension_classifier/testing_images_pred/_predictions.npy" "/data/botml/leaf_dimension_classifier/testing_images/" "/data/botml/leaf_dimension_classifier/classifier_training_testdata/"

#python /data/botml/leaf_dimension_classifier/code/extracting_leaves_cropped_iter_v2.py "/home/botml/euc/models/fb2/pred/_predictions.npy" "/home/botml/euc/models/fb2/test/" "/home/karina/test/"

import numpy as np
import os, json, cv2, random
import torch, torchvision
import detectron2
import pandas as pd
import sys
sys.path.append("/home/botml/code/py")
import model_tools
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("prediction_file", help="Numpy file containing prediction output.", type=str,)
parser.add_argument("image_directory", help="Directory of colored images.", type=str,)
parser.add_argument("output_directory", help="Output directory of crops.", type=str,)
#parser.add_argument("connectivity", type=int, default=4, help="connectivity for connected component analysis")
args = parser.parse_args()

#model_predictions = model_tools.open_predictions(args.prediction_file)
#to see list of images model_predictions.keys()


#image_directory = "/home/botml/euc/models/fb2/test/"
#output_directory = "/home/karina/test/"
#prediction_file = "/home/botml/euc/models/fb2/pred/_predictions.npy"
#model_predictions = model_tools.open_predictions(prediction_file)
connectivity = 4

model_predictions = model_tools.open_predictions(args.prediction_file)
import glob


#package prevents model_tools from opening predictions


for NSWID in (model_predictions):
    NSWIDsplit = os.path.splitext(NSWID)
    raw_im_fil = glob.glob(os.path.join(args.image_directory, NSWID))
    
    for j in enumerate(raw_im_fil):
        raw_im = cv2.imread(j[1])
        
        for i in enumerate(model_predictions[NSWID].pred_masks):
            output_filename_resize = f"{NSWIDsplit[0]}_{str(i[0])}_mask_crop_pad_resize.jpg"
            output_resize_dir = args.output_directory
            rs_img_path = os.path.join(output_resize_dir, output_filename_resize)
            
            array = i[1]
            in_leaf = cv2.imread(j[1])
            in_leaf[~array,:] = [0,0,0]
            
            rows = np.any(in_leaf, axis=1)
            cols = np.any(in_leaf, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            crop_rmin = int(rmin*0.9)
            crop_rmax = int(rmax*1.1)
            crop_cmin = int(cmin*0.9)
            crop_cmax = int(cmax*1.1)

            rsize = crop_rmax - crop_rmin
            csize = crop_cmax - crop_cmin
            
            if rsize >= csize:
                rpad = rsize // 10
                cpad = rpad + (rsize - csize) // 2
            if csize > rsize:
                cpad = csize // 10
                rpad = cpad + (csize - rsize) // 2
        
            gray = cv2.cvtColor(in_leaf, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
            (numLabels, labels, stats, centroid) = output
            mask = np.zeros(array.shape, dtype="uint8")
            area = []
            
            for i in range(1, numLabels):
                A = stats[i, cv2.CC_STAT_AREA]
                area.append(A)
            keepArea = max(area)
            
            for i in range(1, numLabels):
                A = stats[i, cv2.CC_STAT_AREA]		
                if keepArea == A:
                    componentMask = (labels == i).astype("uint8") * 255
                    mask = cv2.bitwise_or(mask, componentMask)
                    mask_inv = cv2.bitwise_not(mask)
        
            res = cv2.bitwise_and(raw_im, raw_im, mask=mask)
            gray_bg = cv2.cvtColor(raw_im, cv2.COLOR_BGR2GRAY)
            background = cv2.bitwise_and(gray_bg, gray_bg, mask = mask_inv)
            background = np.stack((background,)*3, axis=-1)
            img_ca = cv2.add(res, background)
            
            crop_img = img_ca[crop_rmin:crop_rmax, crop_cmin:crop_cmax]
            color = [0, 0, 0]
            pad_img = cv2.copyMakeBorder(crop_img,rpad,rpad,cpad,cpad,cv2.BORDER_CONSTANT,value=color)
            
            rs_img = cv2.resize(pad_img, (500, 500))
            
            print ("Writing", output_filename_resize)
            cv2.imwrite(rs_img_path, rs_img)
