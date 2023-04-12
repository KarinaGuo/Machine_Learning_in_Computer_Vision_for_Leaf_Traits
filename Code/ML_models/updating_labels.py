# python /home/botml/code/py/updating_labels.py /home/botml/euc/models/fb2_vnoUM/temp/ /home/botml/euc/models/fb2_vnoUM/test/ /home/botml/euc/models/fb2_vnoUM/testmap.csv /home/botml/euc/models/fb2_vnoUM/testmap.log

#Create an output.txt to print output

import os, fnmatch, json, sys
directory = str(os.getcwd())

# csv for mapping file
import csv

# arg parse for calling from terminal
import argparse

# parse terminal arguments
parser = argparse.ArgumentParser(description="Converting labels.")
parser.add_argument("label_directory",help="Directory with labels",type=str,)
parser.add_argument("new_directory", help="Directory with new labels.", type=str,)
parser.add_argument("map_list_file", help="Mapping 'old,new' labels.", type=str,)
parser.add_argument("outputs_file", help="track outputs.", type=str,)
args = parser.parse_args()

# putting mappings in a dictionary
mapping_dict = {}
with open(args.map_list_file, mode='r') as inp:
    reader = csv.reader(inp)
    mapping_dict = {rows[0]:rows[1] for rows in reader}

# debug
#change_list = list(mapping_dict.keys())
#print(mapping_dict)
#print(args.label_directory)
#print(args.new_directory)
#print(args.outputs_file)

def findReplace(label_directory, new_directory, outputs_file, mapping_dict):
   change_list = list(mapping_dict.keys())
   for root, dirs, files in os.walk(label_directory):
      for filename in fnmatch.filter(files, '*.json'):  
          jsonpath = os.path.join(label_directory, filename)
          newjsonpath = os.path.join(new_directory, filename)
          #print(jsonpath)
          with open(jsonpath, "r", encoding="utf-8") as read_file:
             maskdata = json.load(read_file)
             if not maskdata["shapes"] or len(maskdata["shapes"]) < 1:
                print("No shapes found")
                return
             shapes = maskdata["shapes"]
             newshapes = shapes
             #print(len(shapes))
             if len(shapes) > 0:
                for i in range(len(shapes)):
                   #print(i)
                   shape = shapes[i]
                   label = shape["label"]
                   if label in change_list :
                      newshapes[i]["label"] = mapping_dict[label]
                      print("file: " + jsonpath + ", Old: " + label + ", New: " + mapping_dict[label])
          newmask=maskdata
          newmask["shapes"] = newshapes
          with open(newjsonpath, "w", encoding="utf-8") as write_file:
            json.dump(newmask, write_file)

                
findReplace(args.label_directory, args.new_directory, args.outputs_file, mapping_dict)
#sys.stdout = open(outputfile, "w")
