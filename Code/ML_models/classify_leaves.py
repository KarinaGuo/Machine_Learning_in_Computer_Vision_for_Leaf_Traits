import sys
sys.path.append("/home/botml/code/py")
import model_tools
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
data_dir   = '/data/botml/leaf_dimension_classifier/leaf_dimension_classifier_L100/classed_training_data/'
val_ratio  = 0.2
num_epochs = 42 
model_out = '/data/botml/leaf_dimension_classifier/leaf_dimension_classifier_L100/model/classifier/model.pth'
model_tools.generic_classifier(data_dir, val_ratio, num_epochs, model_out)
