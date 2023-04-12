import sys
sys.path.append("/home/botml/code/py")
import model_tools

# this required a highly prescribed directory and file structure
# with a directory called training_name containing images and .json files,
# and a file called training_name+'.json'
training_path="/data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/"
training_name="data"
validation_name="validation"
out_dir="model"
out_yaml="model.yaml"
in_yaml="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
in_weights="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
in_yaml_zoo=True
in_weights_zoo=True
ims_per_batch=20
base_lr=0.0001
max_iter=8000
num_classes=1

model_tools.train_val_model(training_path, training_name, validation_name, out_dir, out_yaml, in_yaml, in_weights, in_yaml_zoo, in_weights_zoo, ims_per_batch, base_lr, max_iter, num_classes)


