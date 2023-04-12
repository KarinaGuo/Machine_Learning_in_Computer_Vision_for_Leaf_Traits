import sys
sys.path.append("/home/botml/code/py")
import model_tools

img_dir='/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/test/'
out_dir='/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/pred/'
fext = "jpg"
training_path="/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/"
training_name="data"
yaml_file = "/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/model/model.yaml"
weights_file = "/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/model/model_final.pth"
yaml_zoo=False
weights_zoo=False
num_classes=1
model_tools.visualize_predictions(img_dir, out_dir, fext, training_path, training_name, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7, s1=10, s2=14)
model_tools.predict_in_directory(img_dir, out_dir, fext, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7)

groundtruth_name="test"
#groundtruth_path="/srv/scratch/cornwell/syzygium/image_data/"
model_predictions_file = "/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/pred/_predictions.npy"
thresh_iou = 0
matches_out_file="/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/pred/model_matches.csv"

model_tools.match_groundtruth_prediction(groundtruth_name, training_path, model_predictions_file, thresh_iou, matches_out_file)



### summarize a set of predictions
instances_out_file="/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/pred/model_summary.csv"
model_tools.predictions_summary(model_predictions_file, instances_out_file)




