import os, sys
sys.path.append("/home/botml/code/py")
import model_tools
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
args = parser.parse_args()
base_dir = args.base_dir


model_predictions_file = os.path.join(base_dir, "temp_pred/_predictions.npy")
duplicates_out_file= os.path.join(base_dir, "test_duplicates_out.csv")

model_tools.find_duplicate_predictions(model_predictions_file, duplicates_out_file, 0.7)