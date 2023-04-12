# python cbbv4.py /home/jgb/tmp/dev/data/ /home/jgb/tmp/dev/crop/ --focalbox BB --classes Leaf100 --classes Leaf90

import os
import json
import numpy as np
import cv2
import labelme
import base64
import glob

def boundingBox(points):
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    return min_x, min_y, max_x, max_y

def get_focalbox_by_shape_name(maskdata, focalbox):
   for i, shape in enumerate(maskdata["shapes"]):
      label = shape["label"]
      if label == focalbox:
         points = shape["points"]
         #points = list(map(int, [points[0][0], points[0][1], points[1][0], points[1][1]]))
         #points = list(map(int, [points[0][1], points[0][0], points[1][1], points[1][0]]))
         #points = list(map(int, [points[0][0], points[0][1], points[1][0], points[1][1]]))
         points = list(map(int, boundingBox(points)))
         print("Found focal box")
         #print(points)
         return points

def boundingBox(points):
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    return min_x, min_y, max_x, max_y

def get_new_shapes(maskdata, focalpoints, use_categories):
   shapes = maskdata["shapes"]
   newshapes = []
   #print(len(shapes))
   if len(shapes) > 0:
      for i in range(len(shapes)):
         #print(i)
         shape = shapes[i]
         label = shape["label"]
         #print(use_categories)
#         if label == "LeafLess90" or label == "Leaf90" or label == "Leaf100" :
         if label in use_categories :
           newshape = shape
           points = shape["points"]
           #print(len(points))
           if len(points) > 0:
                bounds = boundingBox(points)
                minX = int(bounds[0])
                minY = int(bounds[1])
                maxX = int(bounds[2]) 
                maxY = int(bounds[3]) 
                infocalbox = True
                if minX < focalpoints[0]:
                   infocalbox = False
                if maxX > focalpoints[2]:
                   infocalbox = False
                if minY < focalpoints[1]:
                   infocalbox = False
                if maxY > focalpoints[3]:
                   infocalbox = False
                if infocalbox:
                   #change contour poitns to fit inside the new bounds
                   newpoints = points
                   for j in range(len(points)):
                       newpoints[j][0] = points[j][0] - focalpoints[0] 
                       newpoints[j][1] = points[j][1] - focalpoints[1]
                   newshape["points"] = newpoints
                   newshapes.append(newshape)
                   # append newshape to newshapes
   return newshapes

def cut_label_files(fileNameNoExtension, jsonPath, imagePath, cutDir, focalbox, use_categories):
    with open(jsonPath, "r", encoding="utf-8") as read_file:
        maskdata = json.load(read_file)
        if not maskdata["shapes"] or len(maskdata["shapes"]) < 1:
            print("No shapes found")
            return

        # get the focal box (ie, a shape with label name in focalbox)
        focalpoints = get_focalbox_by_shape_name(maskdata, focalbox)
        print(focalpoints)
        # focalpoints: xmin, ymin, xmax, ymax

        # if this exists, crop the image, and start setting up new JSON
        image = cv2.imread(imagePath)
        print("Image shape:")
        print(image.shape)
        #cropImg = image[points[0]:points[2], points[1]:points[3]] 
        cropImg = image[focalpoints[1]:focalpoints[3], focalpoints[0]:focalpoints[2]]
        print("Cropped image shape:")
        print(cropImg.shape)
        i=0 # hack
        newImgName = f"{fileNameNoExtension}_{str(i)}.jpg"
        newJSONName = f"{fileNameNoExtension}_{str(i)}.json"
        newImgPath = os.path.join(cutDir, newImgName)
        newJSONPath = os.path.join(cutDir, newJSONName)
        cv2.imwrite(newImgPath, cropImg) 

        imageAsLabelme = labelme.LabelFile.load_image_file(newImgPath)
        imageBase64 = base64.b64encode(imageAsLabelme).decode("utf-8")
        newmask = maskdata
        newmask["imageData"] = imageBase64
        newmask["imagePath"] = newImgName
        newh, neww = cropImg.shape[:2]
        newmask["imageHeight"] = newh
        newmask["imageWidth"] = neww

        newshapes = get_new_shapes(maskdata, focalpoints, use_categories) 
        newmask["shapes"] = newshapes

        with open(newJSONPath, "w", encoding="utf-8") as write_file:
           json.dump(newmask, write_file)

if __name__ == "__main__":
#    workingDir = "/home/jgb/tmp/dev"
# nameNoExtension = "NSW200116"
#    testImg = f"{workingDir}/data/{nameNoExtension}.jpg"
#    testJSON = f"{workingDir}/data/{nameNoExtension}.json"
#    cutDir = f"{workingDir}/out"
#    focalbox='BB'
#    cut_label_files(nameNoExtension, testJSON, testImg, cutDir, focalbox)

   import argparse
   parser = argparse.ArgumentParser(description="Crop labelme json to nominated box.")
   parser.add_argument("label_directory",help="Directory with labels",type=str,)
   parser.add_argument("new_directory", help="Directory for cropped label files.", type=str,)
   parser.add_argument("--focalbox", help="Output json file path.", default="BB")
   parser.add_argument("--classes", action="append", nargs="+", type=str) 
   args = parser.parse_args()
   import itertools 
   arg_classes = list(itertools.chain(*args.classes))
   labelme_json = glob.glob(os.path.join(args.label_directory, "*.json"))
   for num, json_file in enumerate(labelme_json):
      #print(json_file)
      fileName = os.path.basename(json_file)
      fileNamesplit = os.path.splitext(fileName)
      fileNameNoExtension = fileNamesplit[0]
      print("Image " + fileNameNoExtension + "...")
      imageFileName = fileNameNoExtension + ".jpg"
      jpg_file = os.path.join(args.label_directory, imageFileName)
      use_categories=list(itertools.chain(*args.classes))
      cut_label_files(fileNameNoExtension, json_file, jpg_file, args.new_directory, args.focalbox, use_categories)


# perhaps point to a directory, and have it work through all .json files in the dir

# support multiple instances of 'BB" - get the features corresponding to each?

# nominate the set of labels to collect? might be useful since protocol is still evolving, with
# addition of Leaf100B






