import sys
sys.path.append("/home/botml/code/py")
import model_tools
import os

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
parser.add_argument("loop", help="Base directory of processes", type=str,)
args = parser.parse_args()

#Inputs for predict_leaf
if args.loop == "main":
	img_dir= os.path.join(args.base_dir, "temp_image_subset/")
	out_dir= os.path.join(args.base_dir, "temp_pred/")
if args.loop == "model":  
	img_dir= os.path.join(args.base_dir, "input_d2_test/")
	out_dir= os.path.join(args.base_dir, "d2_pred/")
training_path = args.base_dir
training_name= os.path.join(args.base_dir, "trim_d2_images/")
yaml_file = os.path.join(args.base_dir, "model/d2", "model.yaml")
weights_file = os.path.join(args.base_dir, "model/d2", "model_final.pth")

fext = "jpg"
yaml_zoo=False
weights_zoo=False
num_classes=1
#model_tools.visualize_predictions(img_dir, out_dir, fext, training_path, training_name, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7, s1=10, s2=14)
model_tools.predict_in_directory(img_dir, out_dir, fext, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7)
groundtruth_name="test"
model_predictions_file = os.path.join(out_dir, "_predictions.npy")
model_tools.predictions_summary(model_predictions_file)