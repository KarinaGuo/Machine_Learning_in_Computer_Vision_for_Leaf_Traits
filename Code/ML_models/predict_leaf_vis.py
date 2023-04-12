import sys
sys.path.append("/home/botml/code/py")
import model_tools

img_dir='/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/test_QC/'
out_dir='/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/pred_QC/'
fext = "jpg"
training_path="/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/"
training_name="data"
yaml_file = "/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/model/model.yaml"
weights_file = "/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/model/model_final.pth"
yaml_zoo=False
weights_zoo=False
num_classes=1
model_tools.visualize_predictions(img_dir, out_dir, fext, training_path, training_name, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7, s1=10, s2=14)

