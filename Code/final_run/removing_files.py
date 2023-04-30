import glob, os
import csv
import pandas as pd  
import argparse
import fnmatch

parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="", type=str,)
args = parser.parse_args()

directory = os.path.join(args.base_dir, "temp_pred_leaf/subf")
class_pr_res = os.path.join(args.base_dir, "classifier_results_test.csv")
out_res = os.path.join(args.base_dir, "temp_filter.csv")

class_res = pd.read_csv(class_pr_res, sep=',', lineterminator='\n')
filter_res = class_res[class_res.iloc[:,1]=="N"]
filter_res = filter_res.iloc[:,[0]]
filter_res.to_csv(out_res, index = False, encoding='utf8', header=None)

files_to_delete = set()

with open(out_res, 'r') as m:
    m.readline().splitlines()
    for i in m:
        if directory in i[0]:
            i[0] = directory +  i[0].split(directory)[-1]
        if i:
            files_to_delete.add(i.replace(',', os.path.sep))

for i in glob.iglob(os.path.join(directory, "*"), recursive = True):
    print(i)
    for files in files_to_delete:
        try:
          files = files.split('.')
          files_full = os.path.join(directory, files[0] + '.csv')
          os.remove(files_full)
        except:
          pass

for files in glob.iglob(os.path.join(directory, "*.jpg"), recursive = True):
    os.remove(files)