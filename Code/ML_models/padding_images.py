#python /home/botml/code/py/padding_images.py /home/botml/euc/models/leafangle/train/U/ /data/botml/leafangle_classifier/trim/U/ --width 4032 --height 3024

import os
import json
import numpy as np
import cv2
import labelme
import base64
import glob

#what does this line do?
if __name__ == "__main__":

     import argparse

     parser = argparse.ArgumentParser(description="Pad images to nominated size.")
     parser.add_argument("image_directory",help="Directory with images.",type=str,)
     parser.add_argument("new_directory", help="Directory for output images.", type=str,)
     parser.add_argument("--width", type=int,)
     parser.add_argument("--height", type=int,) 
     args = parser.parse_args()

     raw_jpgs = glob.glob(os.path.join(args.image_directory, "*.JPG"))

     for num, raw_jpg in enumerate(raw_jpgs):
          fileName = os.path.basename(raw_jpg)
          fileNamesplit = os.path.splitext(fileName)
          fileNameNoExtension = fileNamesplit[0]
          print("Image " + fileNameNoExtension + "...")
          imageFileName = fileNameNoExtension + ".JPG"
          jpg_file = os.path.join(args.image_directory, imageFileName)

          image = cv2.imread(jpg_file)
          print("Image shape:")
          print(image.shape)
          left = int((args.width - image.shape[1])/2)
          right = left
          top = int((args.height - image.shape[0])/2)
          bottom = top

          color = [0, 0, 0]
          pad_img = cv2.copyMakeBorder (image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 
          newImgName = f"{fileNameNoExtension}_trim.jpg"
          newImgPath = os.path.join(args.new_directory, newImgName)
          cv2.imwrite(newImgPath, pad_img)